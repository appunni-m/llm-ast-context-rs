use clap::Parser;
use serde::Serialize;
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
};
use syn::{
    File as SynFile, Item, ItemFn, ItemMod, ItemStruct, ItemEnum, ItemTrait,
    parse_file, visit::{self, Visit}, Expr, Stmt, Local, Pat,
};
use quote::ToTokens;
use walkdir::WalkDir;


use tiktoken_rs::cl100k_base;
use syn::{UseTree, Type, ItemUse};
use petgraph::Graph;
use std::collections::HashSet;
use syn::{FnArg,PathSegment, ReturnType};
use pathdiff::diff_paths;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Directory to scan for Rust files
    #[arg(default_value = ".", help = "Directory path to parse")]
    directory: String,
    
    /// Output file to store the project context
    #[arg(short, long, default_value = "context.txt", help = "Output file path")]
    output: String,
    
    /// Enable verbose logging
    #[arg(short, long, help = "Enable verbose logging")]
    verbose: bool,
    
    /// Include documentation comments in the output
    #[arg(short, long, help = "Include documentation comments")]
    include_docs: bool,
}

#[derive(Serialize, Clone)]
struct ProjectContext {
    project_root: String,
    files: Vec<FileContext>,
    dependencies: HashMap<String, Vec<String>>,
    type_definitions: HashMap<String, TypeDefinition>,
}

#[derive(Serialize, Clone)]
struct FileContext {
    path: String,
    relative_path: String,
    module_name: String,
    functions: Vec<FunctionInfo>,
    structs: Vec<TypeInfo>,
    enums: Vec<TypeInfo>,
    traits: Vec<TypeInfo>,
    imports: Vec<String>,
    exports: Vec<String>,
    semantic_context: String,
}

#[derive(Serialize, Clone)]
struct FunctionInfo {
    name: String,
    signature: String,
    documentation: Option<String>,
    visibility: String,
    usage: String,
}

#[derive(Serialize, Debug, Clone)]
struct TypeInfo {
    name: String,
    kind: String,
    documentation: Option<String>,
    fields: Vec<String>,
}

#[derive(Serialize, Clone)]
struct TypeDefinition {
    file_path: String,
    definition: String,
    references: Vec<String>,
}

struct AstVisitor {
    functions: Vec<FunctionInfo>,
    structs: Vec<TypeInfo>,
    enums: Vec<TypeInfo>,
    traits: Vec<TypeInfo>,
    imports: Vec<String>,
}

impl<'ast> Visit<'ast> for AstVisitor {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        let doc = node.attrs.iter()
            .filter(|attr| attr.path().is_ident("doc"))
            .map(|attr| attr.to_token_stream().to_string())
            .collect::<Vec<_>>()
            .join("\n");

        let usage = extract_function_usage(node);

        self.functions.push(FunctionInfo {
            name: node.sig.ident.to_string(),
            signature: node.sig.to_token_stream().to_string(),
            documentation: if doc.is_empty() { None } else { Some(doc) },
            visibility: node.vis.to_token_stream().to_string(),
            usage,
        });

        visit::visit_item_fn(self, node);
    }

    fn visit_item_struct(&mut self, node: &'ast ItemStruct) {
        let fields = node.fields
            .iter()
            .map(|f| format!("{}: {}", f.ident.as_ref().unwrap(), f.ty.to_token_stream()))
            .collect();

        self.structs.push(TypeInfo {
            name: node.ident.to_string(),
            kind: "struct".to_string(),
            documentation: extract_doc_comments(&node.attrs),
            fields,
        });

        visit::visit_item_struct(self, node);
    }

    fn visit_item_enum(&mut self, node: &'ast ItemEnum) {
        let fields = node.variants
            .iter()
            .map(|v| v.ident.to_string())
            .collect();

        self.enums.push(TypeInfo {
            name: node.ident.to_string(),
            kind: "enum".to_string(),
            documentation: extract_doc_comments(&node.attrs),
            fields,
        });

        visit::visit_item_enum(self, node);
    }

    fn visit_item_trait(&mut self, node: &'ast ItemTrait) {
        let fields = node.items
            .iter()
            .filter_map(|item| {
                match item {
                    syn::TraitItem::Fn(method) => Some(format!("fn {}", method.sig.ident)),
                    syn::TraitItem::Type(ty) => Some(format!("type {}", ty.ident)),
                    syn::TraitItem::Const(constant) => Some(format!("const {}", constant.ident)),
                    _ => None,
                }
            })
            .collect();

        self.traits.push(TypeInfo {
            name: node.ident.to_string(),
            kind: "trait".to_string(),
            documentation: extract_doc_comments(&node.attrs),
            fields,
        });

        visit::visit_item_trait(self, node);
    }
}

// Add this new struct to track function call context
#[derive(Default)]
struct FunctionCallContext {
    function_calls: Vec<String>,
    call_stack: HashSet<String>,  // Track current call stack to detect recursion
}

impl FunctionCallContext {
    fn new() -> Self {
        Self::default()
    }

    fn add_call(&mut self, function_name: String) {
        if !self.call_stack.contains(&function_name) {
            self.function_calls.push(function_name.clone());
        }
    }

    fn enter_function(&mut self, function_name: &str) -> bool {
        self.call_stack.insert(function_name.to_string())
    }

    fn exit_function(&mut self, function_name: &str) {
        self.call_stack.remove(function_name);
    }
}

fn extract_function_usage(node: &ItemFn) -> String {
    let function_name = node.sig.ident.to_string();
    let mut context = FunctionCallContext::new();

    let params: Vec<String> = node.sig.inputs.iter().map(|param| {
        match param {
            FnArg::Typed(pat_type) => {
                let param_name = match &*pat_type.pat {
                    Pat::Ident(ident) => ident.ident.to_string(),
                    Pat::Wild(_) => "_".to_string(), // Handle wildcard patterns
                    _ => "unknown".to_string(), // Handle other patterns (add more as needed)
                };
                let param_type = format!("{}", quote::quote!(#pat_type.ty));
                format!("{}: {}", param_name, param_type)
            }
            FnArg::Receiver(_) => "self".to_string(), // Handle self parameter
        }
    }).collect();

    let return_type = match &node.sig.output {
        ReturnType::Type(_, ty) => format!("{}", quote::quote!(#ty)),
        ReturnType::Default => "()".to_string(),
    };

    fn traverse_expr(expr: &Expr, context: &mut FunctionCallContext) {
        match expr {
            Expr::Call(call) => {
                if let Expr::Path(path) = &*call.func {
                    let called_fn = path.path.segments.last()
                        .map(|segment: &PathSegment| segment.ident.to_string())
                        .unwrap_or_default();
                    context.add_call(called_fn);
                }
            }
            Expr::MethodCall(method_call) => {
                context.add_call(method_call.method.to_string());
                traverse_expr(&method_call.receiver, context);
                method_call.args.iter().for_each(|arg| traverse_expr(arg, context));
            }
            Expr::Binary(binary) => {
                traverse_expr(&binary.left, context);
                traverse_expr(&binary.right, context);
            }
            Expr::Unary(unary) => traverse_expr(&unary.expr, context),
            Expr::If(if_expr) => {
                traverse_expr(&if_expr.cond, context);
                traverse_block(&if_expr.then_branch, context);
                if let Some(else_branch) = &if_expr.else_branch {
                    traverse_expr(&else_branch.1, context);
                }
            }
            Expr::Loop(loop_expr) => traverse_block(&loop_expr.body, context),
            Expr::While(while_expr) => {
                traverse_expr(&while_expr.cond, context);
                traverse_block(&while_expr.body, context);
            }
            Expr::ForLoop(for_loop) => traverse_block(&for_loop.body, context),
            Expr::Match(match_expr) => {
                traverse_expr(&match_expr.expr, context);
                for arm in &match_expr.arms {
                    if let Some(guard) = &arm.guard {
                        traverse_expr(&guard.1, context);
                    }
                    traverse_expr(&arm.body, context);
                }
            }
            Expr::Closure(closure) => {
                if let syn::Expr::Block(block) = &*closure.body {
                    traverse_block(&block.block, context);
                }
            }
            _ => { /* Handle other Expr variants as needed */ }
        }
    }

    fn traverse_block(block: &syn::Block, context: &mut FunctionCallContext) {
        for stmt in &block.stmts {
            traverse_stmt(stmt, context);
        }
    }

    fn traverse_stmt(stmt: &Stmt, context: &mut FunctionCallContext) {
        match stmt {
            Stmt::Expr(expr, _) => traverse_expr(expr, context),
            Stmt::Local(Local { init: Some(init), .. }) => {
                traverse_expr(&init.expr, context);
            }
            Stmt::Item(_) => {}, // Ignore Item statements
            _ => {}
        }
    }

    // Enter the function context before analysis
    context.enter_function(&function_name);
    traverse_block(&*node.block, &mut context);
    context.exit_function(&function_name);

    format!(
        "Function: {}\nParameters: {}\nReturn Type: {}\nCalled Functions: {} (Recursive: {})",
        function_name,
        params.join(", "),
        return_type,
        context.function_calls.join(", "),
        context.call_stack.contains(&function_name)
    )
}

fn extract_doc_comments(attrs: &[syn::Attribute]) -> Option<String> {
    let docs: Vec<String> = attrs.iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .filter_map(|attr| {
            if let syn::Meta::NameValue(meta) = &attr.meta {
                if let syn::Expr::Lit(lit_expr) = &meta.value {
                    if let syn::Lit::Str(lit_str) = &lit_expr.lit {
                        return Some(lit_str.value());
                    }
                }
            }
            None
        })
        .collect();

    if docs.is_empty() {
        None
    } else {
        Some(docs.join("\n"))
    }
}

fn main() -> Result<(), io::Error> {
    let args = Args::parse();
    println!("Analyzing project in directory: {}", args.directory);

    let rust_files = collect_rust_files(&args.directory, args.verbose)?;
    println!("Found {} Rust files to analyze.", rust_files.len());

    let mut project_context = ProjectContext {
        project_root: args.directory.clone(),
        files: Vec::new(),
        dependencies: HashMap::new(),
        type_definitions: HashMap::new(),
    };

    let mut failed_files = Vec::new();

    for file_path in rust_files {
        if args.verbose {
            println!("Processing: {}", file_path);
        }

        match analyze_file(&file_path, &args) {
            Ok(file_context) => {
                // Update type definitions map
                for type_info in &file_context.structs {
                    project_context.type_definitions.insert(
                        type_info.name.clone(),
                        TypeDefinition {
                            file_path: file_path.clone(),
                            definition: format!("{:?}", type_info),
                            references: Vec::new(),
                        },
                    );
                }
                project_context.files.push(file_context);
            }
            Err(e) => {
                failed_files.push((file_path.clone(), e.to_string()));
                if args.verbose {
                    eprintln!("Failed to analyze {}: {}", file_path, e);
                }
            }
        }
    }

    // Print summary of failures at the end if any
    if !failed_files.is_empty() {
        println!("\nAnalysis completed with {} failures:", failed_files.len());
        if args.verbose {
            for (file, error) in failed_files {
                println!("- {}: {}", file, error);
            }
        }
    }

    // Second pass: Analyze dependencies and references
    analyze_dependencies(&mut project_context);

    // Create output file and generate LLM context
    let mut output_file = File::create(&args.output)?;
    let llm_context = project_context.to_llm_context();
    
    // Write the context in a readable text format
    writeln!(output_file, "# Project Analysis\n")?;
    
    // Write project overview
    writeln!(output_file, "## Project Overview")?;
    writeln!(output_file, "Root Directory: {}", llm_context.project_overview.root)?;
    writeln!(output_file, "Total Files: {}", llm_context.project_overview.total_files)?;
    writeln!(output_file, "Global Types: {}\n", llm_context.project_overview.global_types.join(", "))?;
    
    // Write dependency information with better organization
    writeln!(output_file, "## Dependencies\n")?;
    
    // Group dependencies by source file and clean up paths
    let mut file_deps: HashMap<String, HashSet<String>> = HashMap::new();
    for edge in &llm_context.dependencies.edges {
        let from_path = PathBuf::from(&edge.from);
        let root_path = PathBuf::from(&llm_context.project_overview.root);
        
        let from = diff_paths(&from_path, &root_path)
            .unwrap_or_else(|| from_path.clone())
            .to_string_lossy()
            .to_string();
        
        file_deps.entry(from)
            .or_default()
            .insert(edge.to.clone());
    }

    // Write organized dependencies
    for (file, deps) in &file_deps {
        if !deps.is_empty() {
            writeln!(output_file, "{} depends on:", file)?;
            // Sort dependencies for consistent output
            let mut sorted_deps: Vec<_> = deps.iter().collect();
            sorted_deps.sort();
            for dep in sorted_deps {
                writeln!(output_file, "- {}", dep)?;
            }
            writeln!(output_file)?;
        }
    }
    
    // Write type references, excluding self-references
    writeln!(output_file, "\n## Type References")?;
    for (type_name, references) in &llm_context.dependencies.type_references {
        let external_refs: Vec<_> = references.iter()
            .filter(|&ref_path| {
                // Get the file where the type is defined
                let binding = String::new();
                let type_file = project_context.type_definitions
                    .get(type_name)
                    .map(|def| &def.file_path)
                    .unwrap_or(&binding);
                // Only include references from other files
                ref_path != type_file
            })
            .collect();

        if !external_refs.is_empty() {
            writeln!(output_file, "\n{} is used in:", type_name)?;
            for reference in external_refs {
                // Use only filename instead of full path
                let file_name = PathBuf::from(reference)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                writeln!(output_file, "- {}", file_name)?;
            }
        }
    }
    
    // Write detailed file analysis
    writeln!(output_file, "\n## File Analysis\n")?;
    for file in &llm_context.files {
        writeln!(output_file, "---\n{}", file.semantic_context)?;
    }
    
    writeln!(output_file, "\nTotal Tokens: {}", llm_context.total_tokens)?;

    println!("Project analysis completed. Context saved to {}", args.output);
    Ok(())
}

fn collect_rust_files(directory: &str, verbose: bool) -> io::Result<Vec<String>> {
    let absolute_path = fs::canonicalize(directory)?;
    let mut files = Vec::new();
    
    // Add paths to skip
    let skip_paths = ["target", "docs", ".git", "examples", "tests"];
    
    for entry in WalkDir::new(&absolute_path)
        .follow_links(true)
        .into_iter()
        .filter_entry(|e| {
            let path = e.path();
            // Skip directories we don't want to analyze
            !skip_paths.iter().any(|skip| {
                path.components().any(|c| {
                    c.as_os_str().to_string_lossy() == *skip
                })
            }) && 
            // Skip hidden files/dirs
            !is_hidden_or_target(path)
        })
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        
        if path.is_file() && path.extension().map_or(false, |ext| ext == "rs") {
            // Skip files that are too small (likely incomplete)
            if fs::metadata(path).map(|m| m.len()).unwrap_or(0) < 10 {
                if verbose {
                    println!("Skipping small file: {}", path.display());
                }
                continue;
            }
            
            if verbose {
                println!("Found Rust file: {}", path.display());
            }
            if let Some(path_str) = path.to_str() {
                files.push(path_str.to_string());
            }
        }
    }

    if verbose {
        println!("Total valid Rust files found: {}", files.len());
    }

    Ok(files)
}

fn is_hidden_or_target(path: &Path) -> bool {
    path.components().any(|comp| {
        let name = comp.as_os_str().to_string_lossy();
        name.starts_with('.') || name == "target"
    })
}
fn analyze_file(file_path: &str, args: &Args) -> Result<FileContext, syn::Error> {
    let source = fs::read_to_string(file_path).map_err(|e| syn::Error::new_spanned(&proc_macro2::TokenStream::new(), e.to_string()))?;
    let ast = parse_file(&source)?;

    let mut visitor = AstVisitor {
        functions: Vec::new(),
        structs: Vec::new(),
        enums: Vec::new(),
        traits: Vec::new(),
        imports: Vec::new(),
    };
    
    visitor.visit_file(&ast);
    
    // Get just the path relative to src/ directory
    let path = PathBuf::from(file_path);
    let relative_path = if let Ok(rel) = path.strip_prefix(&args.directory) {
        rel.to_string_lossy().trim_start_matches('/').to_string()
    } else {
        // If strip_prefix fails, try to find "src" in the path
        path.components()
            .skip_while(|c| c.as_os_str() != "src")
            .collect::<PathBuf>()
            .to_string_lossy()
            .to_string()
    };

    Ok(FileContext {
        path: file_path.to_string(),  // Keep full path for internal use
        relative_path,                // Use cleaned relative path
        module_name: PathBuf::from(file_path)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        functions: visitor.functions,
        structs: visitor.structs,
        enums: visitor.enums,
        traits: visitor.traits,
        imports: visitor.imports,
        exports: Vec::new(),
        semantic_context: String::new(),
    })
}

#[derive(Default)]
struct DependencyVisitor {
    uses: HashSet<String>,
    type_references: HashSet<String>,
}

impl<'ast> Visit<'ast> for DependencyVisitor {
    fn visit_item_use(&mut self, node: &'ast ItemUse) {
        self.extract_use_path(&node.tree);
        visit::visit_item_use(self, node);
    }

    fn visit_type(&mut self, node: &'ast Type) {
        if let Type::Path(type_path) = node {
            if let Some(segment) = type_path.path.segments.first() {
                self.type_references.insert(segment.ident.to_string());
            }
        }
        visit::visit_type(self, node);
    }
}

impl DependencyVisitor {
    fn extract_use_path(&mut self, tree: &UseTree) {
        match tree {
            UseTree::Path(path) => {
                // Only take external crate dependencies
                let crate_name = path.ident.to_string();
                if !is_internal_dep(&crate_name) && !is_std_dep(&crate_name) {
                    self.uses.insert(crate_name);
                }
            }
            UseTree::Name(name) => {
                let crate_name = name.ident.to_string();
                if !is_internal_dep(&crate_name) && !is_std_dep(&crate_name) {
                    self.uses.insert(crate_name);
                }
            }
            UseTree::Group(group) => {
                for tree in &group.items {
                    self.extract_use_path(tree);
                }
            }
            _ => {} // Skip other import types
        }
    }
}

fn is_internal_dep(dep: &str) -> bool {
    dep == "crate" || dep == "super" || dep == "self"
}

// Add this function to find the common path prefix
fn find_common_prefix(paths: &[String]) -> PathBuf {
    if paths.is_empty() {
        return PathBuf::new();
    }

    let mut common = PathBuf::from(&paths[0]);
    for path in paths.iter().skip(1) {
        let mut new_common = PathBuf::new();
        let path = PathBuf::from(path);
        
        for (c1, c2) in common.components().zip(path.components()) {
            if c1 == c2 {
                new_common.push(c1);
            } else {
                break;
            }
        }
        common = new_common;
        if common.as_os_str().is_empty() {
            break;
        }
    }
    common
}

// Modify the analyze_dependencies function
fn analyze_dependencies(project_context: &mut ProjectContext) {
    // First find common prefix of all file paths
    let all_paths: Vec<String> = project_context.files.iter()
        .map(|f| f.path.clone())
        .collect();
    let common_prefix = find_common_prefix(&all_paths);

    let mut dependency_graph = Graph::<String, ()>::new();
    let mut node_indices = HashMap::new();

    // Create nodes for all files with simplified paths
    for file in &project_context.files {
        let path = PathBuf::from(&file.path);
        let relative_path = path.strip_prefix(&common_prefix)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();
            
        let idx = dependency_graph.add_node(relative_path.clone());
        node_indices.insert(file.path.clone(), idx);
    }

    // Rest of the function using the simplified paths...
}

// Also modify the main function's dependency writing
fn write_dependencies(output_file: &mut File, llm_context: &LLMContext) -> io::Result<()> {
    writeln!(output_file, "## Dependencies\n")?;
    
    // Get all paths and find common prefix to strip
    let all_paths: Vec<String> = llm_context.dependencies.edges.iter()
        .map(|e| e.from.clone())
        .collect();
    
    let common_prefix = if let Some(first) = all_paths.first() {
        let mut prefix = PathBuf::from(first);
        while let Some(parent) = prefix.parent() {
            if all_paths.iter().all(|p| p.starts_with(prefix.to_str().unwrap_or(""))) {
                prefix = parent.to_path_buf();
            } else {
                break;
            }
        }
        prefix
    } else {
        PathBuf::new()
    };

    // Group dependencies by file
    let mut file_deps: HashMap<String, HashSet<String>> = HashMap::new();
    for edge in &llm_context.dependencies.edges {
        let from_path = PathBuf::from(&edge.from);
        // Get path relative to src/
        let relative_path = from_path
            .strip_prefix(&common_prefix)
            .unwrap_or(&from_path)
            .to_str()
            .unwrap_or("")
            .trim_start_matches("/")
            .to_string();

        // Only include external crate dependencies
        if !is_internal_dep(&edge.to) && !is_std_dep(&edge.to) {
            file_deps.entry(relative_path)
                .or_default()
                .insert(edge.to.clone());
        }
    }

    // Sort files for consistent output
    let mut files: Vec<_> = file_deps.keys().collect();
    files.sort();

    // Write organized dependencies
    for file in files {
        if let Some(deps) = file_deps.get(file) {
            if !deps.is_empty() {
                writeln!(output_file, "{} depends on:", file)?;
                let mut sorted_deps: Vec<_> = deps.iter().collect();
                sorted_deps.sort();
                for dep in sorted_deps {
                    writeln!(output_file, "- {}", dep)?;
                }
                writeln!(output_file)?;
            }
        }
    }

    Ok(())
}

// Helper function to identify standard library dependencies
fn is_std_dep(dep: &str) -> bool {
    let std_modules = ["std", "core", "alloc", "proc_macro"];
    std_modules.contains(&dep) || dep.starts_with("std::") || dep.starts_with("core::")
}

#[derive(Serialize)]
pub struct LLMContext {
    project_overview: ProjectOverview,
    files: Vec<FileContext>,
    dependencies: DependencyGraph,
    total_tokens: usize,
}

#[derive(Serialize)]
pub struct ProjectOverview {
    root: String,
    total_files: usize,
    global_types: Vec<String>,
}

#[derive(Serialize)]
pub struct DependencyGraph {
    nodes: Vec<String>, // File paths
    edges: Vec<DependencyEdge>,
    type_references: HashMap<String, Vec<String>>, // type -> [files using it]
}

#[derive(Serialize)]
pub struct DependencyEdge {
    from: String,
    to: String,
    imports: Vec<String>,
}

impl ProjectContext {
    fn to_llm_context(&self) -> LLMContext {
        let mut semantic_context = String::new();

        // Process each file's semantic context
        let files: Vec<FileContext> = self.files.iter().map(|file| {
            let mut semantic_context = String::new();

            // Show both module name and relative path
            semantic_context.push_str(&format!("# {}\n", file.module_name));
            semantic_context.push_str(&format!("Path: {}\n\n", file.relative_path));

            // Group types by their role in the system
            let mut data_types = Vec::new();
            let mut service_types = Vec::new();
            let mut utility_types = Vec::new();

            // Categorize types based on their usage and fields
            for type_info in &file.structs {
                let type_summary = format!("### {}\n{}\n",
                    type_info.name,
                    type_info.fields.iter()
                        .map(|f| f.replace(" < ", "<").replace(" > ", ">"))
                        .collect::<Vec<_>>()
                        .join(", ")
                );

                if type_info.name.ends_with("Context") || type_info.name.ends_with("Info") {
                    data_types.push(type_summary);
                } else if type_info.name.ends_with("Visitor") || type_info.name.contains("Service") {
                    service_types.push(type_summary);
                } else {
                    utility_types.push(type_summary);
                }
            }

            // Write categorized types
            if !data_types.is_empty() {
                semantic_context.push_str("## Data Types\n");
                data_types.iter().for_each(|t| semantic_context.push_str(t));
            }
            if !service_types.is_empty() {
                semantic_context.push_str("## Service Types\n");
                service_types.iter().for_each(|t| semantic_context.push_str(t));
            }
            if !utility_types.is_empty() {
                semantic_context.push_str("## Utility Types\n");
                utility_types.iter().for_each(|t| semantic_context.push_str(t));
            }

            // Group functions by their role
            if !file.functions.is_empty() {
                semantic_context.push_str("## Functions\n\n");
                
                // Only show functions that have meaningful interactions
                for func in &file.functions {
                    if let Some(usage) = extract_meaningful_usage(&func.usage) {
                        semantic_context.push_str(&format!("### {}\n", func.name));
                        if !usage.is_empty() {
                            semantic_context.push_str(&format!("{}\n", usage));
                        }
                    }
                }
            }

            FileContext {
                path: file.path.clone(),
                relative_path: file.relative_path.clone(),
                module_name: file.module_name.clone(),
                functions: file.functions.clone(),
                structs: file.structs.clone(),
                enums: file.enums.clone(),
                traits: file.traits.clone(),
                imports: file.imports.clone(),
                exports: file.exports.clone(),
                semantic_context,
            }
        }).collect();

        // Create project overview
        let project_overview = ProjectOverview {
            root: self.project_root.clone(),
            total_files: self.files.len(),
            global_types: self.type_definitions.keys().cloned().collect(),
        };

        // Build dependency graph
        let mut dependency_graph = DependencyGraph {
            nodes: self.files.iter().map(|f| f.path.clone()).collect(),
            edges: Vec::new(),
            type_references: HashMap::new(),
        };

        // Add dependency edges
        for file in &self.files {
            if let Some(deps) = self.dependencies.get(&file.path) {
                for dep in deps {
                    dependency_graph.edges.push(DependencyEdge {
                        from: file.path.clone(),
                        to: dep.clone(),
                        imports: file.imports.clone(),
                    });
                }
            }
        }

        // Add type references
        for (type_name, type_def) in &self.type_definitions {
            dependency_graph.type_references.insert(
                type_name.clone(),
                type_def.references.clone(),
            );
        }

        LLMContext {
            project_overview,
            files,
            dependencies: dependency_graph,
            total_tokens: 0,
        }
    }
}

// Improve the meaningful usage extraction
fn extract_meaningful_usage(usage: &str) -> Option<String> {
    let lines: Vec<&str> = usage.lines().collect();
    let mut meaningful_info = Vec::new();
    let mut method_context = String::new();

    for line in lines {
        if line.starts_with("Function:") {
            method_context = line["Function:".len()..].trim().to_string();
        } else if line.starts_with("Parameters:") {
            // Only show parameters if they're meaningful (not just basic types)
            let params = line["Parameters:".len()..].trim();
            if !params.is_empty() && !params.contains("&str") && !params.contains("String") {
                meaningful_info.push(format!("📝 {}", params));
            }
        } else if line.starts_with("Return Type:") {
            // Only show return type if it's not unit or basic types
            let ret_type = line["Return Type:".len()..].trim();
            if ret_type != "()" && !ret_type.contains("String") {
                meaningful_info.push(format!("↩️ {}", ret_type));
            }
        } else if line.starts_with("Called Functions:") {
            let calls: Vec<&str> = line["Called Functions:".len()..]
                .split(',')
                .map(str::trim)
                .filter(|&call| {
                    !["to_string", "clone", "new", "default", "unwrap", "map", 
                      "collect", "iter", "into_iter", "push", "insert", "is_empty",
                      "contains", "split", "join", "format", "write", "read"]
                        .contains(&call)
                })
                .collect();
            
            if !calls.is_empty() {
                // Group related calls together
                let mut grouped_calls = Vec::<String>::new();
                let mut current_group = Vec::<String>::new();
                
                for call in calls {
                    if current_group.is_empty() || are_related_calls(&current_group.last().unwrap(), call) {
                        current_group.push(call.to_string());
                    } else {
                        grouped_calls.push(current_group.join(" → "));
                        current_group = vec![call.to_string()];
                    }
                }
                if !current_group.is_empty() {
                    grouped_calls.push(current_group.join(" → "));
                }
                
                meaningful_info.push(format!("🔄 {}", grouped_calls.join(" | ")));
            }
        }
    }

    if meaningful_info.is_empty() {
        None
    } else {
        Some(format!("{}\n{}", method_context, meaningful_info.join("\n")))
    }
}

fn are_related_calls(prev: &str, current: &str) -> bool {
    // Group calls that are likely part of the same operation
    let related_prefixes = [
        ("parse", "validate"),
        ("get", "set"),
        ("read", "write"),
        ("create", "delete"),
        ("open", "close"),
        ("start", "stop"),
    ];

    for (prefix1, prefix2) in related_prefixes {
        if (prev.starts_with(prefix1) && current.starts_with(prefix2)) ||
           (prev.starts_with(prefix2) && current.starts_with(prefix1)) {
            return true;
        }
    }

    false
}
