// Local diagnostic (not committed): the program and layout the translator infers from an FSPG part.
fn main() {
    for path in std::env::args().skip(1) {
        let d = std::fs::read(&path).unwrap();
        let n = u32::from_le_bytes(d[28..32].try_into().unwrap()) as usize;
        for i in 0..n {
            let off = u32::from_le_bytes(d[32 + 4 * i..36 + 4 * i].try_into().unwrap()) as usize;
            if &d[off..off + 4] == b"ACCC" {
                let len = u32::from_le_bytes(d[off + 4..off + 8].try_into().unwrap()) as usize;
                match victory_shader::emit::graphics::program(&d[off + 8..off + 8 + len]) {
                    Ok(p) => println!("{}: stage {:?} layout {:?}", path.rsplit('/').next().unwrap(), p.stage, p.layout),
                    Err(e) => println!("{e}"),
                }
            }
        }
    }
}
