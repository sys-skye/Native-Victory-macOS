// The part of the workspace's `Cargo.lock` the translator is built from, for
// `VICTORY_SHADER_SOURCE_HASH`. Included by `build.rs` and by `src/lock_tests.rs`.

/// The lock entries (`name version source checksum`, sorted) of every package `root` depends
/// on, directly or not; `root`'s own entry is left out (its sources are hashed whole). A
/// package outside that set, an app's or a test tool's, changes nothing here.
pub fn dependency_entries(lock: &str, root: &str) -> Vec<String> {
    #[derive(Default)]
    struct Package {
        name: String,
        version: String,
        rest: String,
        dependencies: Vec<String>,
    }
    let mut packages: Vec<Package> = Vec::new();
    let mut in_dependencies = false;
    for line in lock.lines().map(str::trim) {
        if line == "[[package]]" {
            packages.push(Package::default());
            in_dependencies = false;
            continue;
        }
        let Some(p) = packages.last_mut() else { continue };
        if in_dependencies {
            if line.starts_with(']') {
                in_dependencies = false;
            } else {
                p.dependencies.push(line.trim_end_matches(',').trim_matches('"').to_owned());
            }
            continue;
        }
        let Some((key, value)) = line.split_once(" = ") else { continue };
        let value = value.trim_matches('"');
        match key {
            "name" => p.name = value.to_owned(),
            "version" => p.version = value.to_owned(),
            "source" | "checksum" => {
                p.rest.push(' ');
                p.rest.push_str(value);
            }
            "dependencies" => {
                // `dependencies = [` then one per line, or `[]`.
                in_dependencies = !value.ends_with(']');
            }
            _ => {}
        }
    }
    // A dependency is `name`, or `name version` where the lock holds several versions of it.
    let find = |dependency: &str| {
        let mut words = dependency.split_whitespace();
        let name = words.next().unwrap_or_default();
        let version = words.next();
        packages
            .iter()
            .position(|p| p.name == name && version.is_none_or(|v| p.version == v))
    };
    let mut seen = vec![false; packages.len()];
    let mut todo: Vec<usize> = packages.iter().position(|p| p.name == root).into_iter().collect();
    if let Some(&at) = todo.first() {
        seen[at] = true;
    }
    let mut entries = Vec::new();
    while let Some(at) = todo.pop() {
        for dependency in &packages[at].dependencies {
            if let Some(next) = find(dependency)
                && !std::mem::replace(&mut seen[next], true)
            {
                let p = &packages[next];
                entries.push(format!("{} {}{}", p.name, p.version, p.rest));
                todo.push(next);
            }
        }
    }
    entries.sort();
    entries
}
