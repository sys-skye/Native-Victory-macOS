// Local diagnostic (not committed): translates dumped containers with the native host's options.
// `extra_<hash>_<rect>_<extra hex>_<copy hex>.bin` also carries a guessed draw's extra outputs.
use victory_shader::emit::{Options, translate_with};
fn main() {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).expect("read");
        let mut o = Options { wave32_host: true, zero_copy_memory: true, ..Options::default() };
        let name = path.rsplit('/').next().unwrap_or(&path);
        if let Some(rest) = name.strip_prefix("extra_").and_then(|r| r.strip_suffix(".bin")) {
            let f: Vec<&str> = rest.split('_').collect();
            o.rect_list = f[1] == "1";
            o.extra_params = u32::from_str_radix(f[2], 16).expect("extra");
            for (i, c) in o.copy_params.iter_mut().enumerate() {
                *c = u8::from_str_radix(&f[3][2 * i..2 * i + 2], 16).expect("copy");
            }
        }
        match translate_with(&bytes, o) {
            Ok(_) => println!("{name}: OK"),
            Err(e) => println!("{name}: {e}"),
        }
    }
}
