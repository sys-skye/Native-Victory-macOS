//! Pins for the control-flow algorithms of `ir.rs` on hand-built block
//! graphs: `reverse_postorder`, `idoms` and `irreducible_edge`.

use crate::ir::{self, Block, Exit};

fn block(exit: Exit) -> Block {
    Block {
        offset: 0,
        preds: Vec::new(),
        phis: Vec::new(),
        insts: Vec::new(),
        exit,
    }
}

/// Fills in `preds` from the exits, as the builder's `link` does.
fn link(mut blocks: Vec<Block>) -> Vec<Block> {
    for b in blocks.iter_mut() {
        b.preds.clear();
    }
    for b in 0..blocks.len() {
        for s in blocks[b].exit.successors() {
            blocks[s].preds.push(b);
        }
    }
    blocks
}

fn order_of(blocks: &[Block]) -> (Vec<usize>, Vec<usize>) {
    let order = ir::reverse_postorder(blocks);
    let mut index = vec![usize::MAX; blocks.len()];
    for (i, &b) in order.iter().enumerate() {
        index[b] = i;
    }
    (order, index)
}

/// 0 -> {1, 2}, 1 -> 3, 2 -> 3, 3 -> end: a diamond.
fn diamond() -> Vec<Block> {
    vec![
        block(Exit::Branch {
            taken: 1,
            not_taken: 2,
        }),
        block(Exit::Jump(3)),
        block(Exit::Jump(3)),
        block(Exit::End),
    ]
}

#[test]
fn reverse_postorder_starts_at_the_entry_and_visits_each_block_once() {
    let blocks = link(diamond());
    let (order, index) = order_of(&blocks);
    assert_eq!(order, [0, 2, 1, 3], "the not-taken arm is explored first");
    assert_eq!(index[0], 0);
    assert!(order.windows(2).all(|w| index[w[0]] < index[w[1]]));
}

#[test]
fn idoms_of_a_diamond_are_all_the_entry() {
    let blocks = link(diamond());
    let (order, index) = order_of(&blocks);
    assert_eq!(ir::idoms(&blocks, &order, &index), [0, 0, 0, 0]);
}

/// 0 -> 1, 1 -> {2, 3}, 2 -> 1 (back edge), 3 -> end: a reducible loop.
fn looped() -> Vec<Block> {
    vec![
        block(Exit::Jump(1)),
        block(Exit::Branch {
            taken: 2,
            not_taken: 3,
        }),
        block(Exit::Jump(1)),
        block(Exit::End),
    ]
}

#[test]
fn idoms_of_a_loop_are_the_header() {
    let blocks = link(looped());
    let (order, index) = order_of(&blocks);
    let idom = ir::idoms(&blocks, &order, &index);
    assert_eq!(idom[0], 0, "the entry dominates itself");
    assert_eq!(idom[1], 0);
    assert_eq!(
        idom[2], 1,
        "the back-edge source is dominated by the header"
    );
    assert_eq!(idom[3], 1);
}

#[test]
fn a_reducible_loop_has_no_irreducible_edge() {
    let blocks = link(looped());
    let (order, index) = order_of(&blocks);
    assert_eq!(ir::irreducible_edge(&blocks, &order, &index), None);
}

#[test]
fn an_edge_into_a_loop_the_target_does_not_dominate_is_irreducible() {
    // 0 -> {1, 2}, 1 -> 3, 2 -> 3, 3 -> {1, 4}, 4 -> end: 3 jumps back to 1,
    // but 1 is also entered from 0, so 3 does not dominate 1.
    let blocks = link(vec![
        block(Exit::Branch {
            taken: 1,
            not_taken: 2,
        }),
        block(Exit::Jump(3)),
        block(Exit::Jump(3)),
        block(Exit::Branch {
            taken: 1,
            not_taken: 4,
        }),
        block(Exit::End),
    ]);
    let (order, index) = order_of(&blocks);
    assert_eq!(ir::irreducible_edge(&blocks, &order, &index), Some((3, 1)));
}
