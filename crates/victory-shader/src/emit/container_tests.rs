//! Pins for the PC-container writer (`container::build`) and the signature
//! serializer (`container::signature`), which the DXBC reader and the shader
//! cache rely on.

use super::container::{self, SigElement};
use super::signature as parse_signature;
use crate::dxbc::{Container, FourCC};

fn el(name: &str, index: u32, kind: u8, row: Option<u32>, usage: u8) -> SigElement {
    SigElement {
        name: name.into(),
        index,
        kind,
        comp: 9, // f32
        interp: 0,
        row,
        col: 0,
        cols: 4,
        usage,
        half: false,
    }
}

#[test]
fn built_containers_parse_back_with_their_parts_in_order() {
    let parts: &[(&[u8; 4], &[u8])] = &[
        (b"ISG1", &[1, 2, 3, 4]),
        (b"DXIL", &[5; 100]),
        (b"OSG1", &[9]),
    ];
    let bytes = container::build(parts);
    assert_eq!(&bytes[..4], b"DXBC");
    let c = Container::parse(&bytes).expect("parses");
    assert_eq!(c.version, (1, 0));
    assert_eq!(c.parts.len(), 3);
    let fourccs: Vec<_> = c.parts.iter().map(|p| p.fourcc).collect();
    assert_eq!(
        fourccs,
        [FourCC(*b"ISG1"), FourCC(*b"DXIL"), FourCC(*b"OSG1")]
    );
    assert_eq!(c.parts[0].data, &[1, 2, 3, 4]);
    assert_eq!(c.parts[1].data, &[5; 100]);
    // Part offsets start after the header and the offset table.
    assert_eq!(c.parts[0].offset, 0x20 + 4 * 3);
    assert_eq!(c.parts[1].offset, 0x20 + 4 * 3 + 8 + 4);
    assert_eq!(c.part(FourCC(*b"DXIL")).unwrap().data.len(), 100);
}

#[test]
fn the_digest_is_the_hash_of_everything_after_the_header() {
    let bytes = container::build(&[(b"DXIL", b"program")]);
    assert_eq!(c_digest(&bytes), container::hash(&bytes[20..]));
    // One changed part byte changes the digest, and the parse still reads it.
    let other = container::build(&[(b"DXIL", b"progral")]);
    assert_ne!(c_digest(&bytes), c_digest(&other));
    fn c_digest(b: &[u8]) -> [u8; 16] {
        b[4..20].try_into().unwrap()
    }
}

#[test]
fn an_empty_container_parses() {
    let bytes = container::build(&[]);
    assert_eq!(bytes.len(), 0x20);
    let c = Container::parse(&bytes).expect("parses");
    assert!(c.parts.is_empty());
}

#[test]
fn the_signature_serializes_header_elements_then_names() {
    // Rows are written in register order; a system value's name is shared by kind.
    let elements = [
        el("TEXCOORD", 2, 0, Some(1), 0b1111),
        el("SV_Position", 0, 3, Some(0), 0b1111),
    ];
    let bytes = container::signature(&elements, true);
    assert_eq!(u32(&bytes, 0), 2, "element count");
    assert_eq!(u32(&bytes, 4), 8, "elements start after the header");
    // Element 0 is SV_Position (row 0 first): stream, name offset, index,
    // system value, component type, row.
    let at = 8;
    assert_eq!(u32(&bytes, at), 0, "stream");
    assert_eq!(u32(&bytes, at + 12), 1, "SV_Position's system value");
    assert_eq!(u32(&bytes, at + 20), 0, "row");
    assert_eq!(bytes[at + 24], 0b1111, "mask");
    assert_eq!(bytes[at + 25], 0b1111, "read-write mask of an input");
    // Then TEXCOORD, then the two names NUL-terminated.
    let at = 8 + 32;
    assert_eq!(u32(&bytes, at + 8), 2, "index");
    assert_eq!(u32(&bytes, at + 12), 0, "arbitrary semantic");
    assert_eq!(u32(&bytes, at + 20), 1, "row");
    let names = &bytes[8 + 64..];
    assert_eq!(
        names, b"TEXCOORD\0SV_Position\0",
        "names in element order, not row order"
    );
    fn u32(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
    }
}

#[test]
fn a_serialized_signature_parses_back() {
    let elements = [
        el("SV_Position", 0, 3, Some(0), 0b1111),
        el("TEXCOORD", 2, 0, Some(1), 0b1010),
    ];
    let bytes = container::signature(&elements, true);
    let parsed = parse_signature(&bytes).expect("parses");
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].name, "SV_Position");
    assert_eq!(parsed[0].kind, 3);
    assert_eq!(parsed[0].row, Some(0));
    assert_eq!(parsed[1].name, "TEXCOORD");
    assert_eq!(parsed[1].index, 2);
    assert_eq!(parsed[1].kind, 0);
    assert_eq!(parsed[1].row, Some(1));
    // The parser derives col/cols from the mask and leaves usage 0 (the
    // read-write byte's bits are applied by the caller).
    assert_eq!((parsed[1].col, parsed[1].cols, parsed[1].usage), (0, 4, 0));
}
