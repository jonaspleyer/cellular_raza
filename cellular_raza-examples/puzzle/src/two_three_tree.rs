struct LeafNode<K, V> {
    key: K,
    value: V,
}

enum INode<'a, K, V> {
    TwoNode {
        field: Box<&'a K>,
        children: [Box<Node<'a, K, V>>; 2],
    },
    ThreeNode {
        fields: [&'a K; 2],
        children: [Box<Node<'a, K, V>>; 3],
    }
}

enum Node<'a, K, V> {
    Leaf(LeafNode<K, V>),
    Interior(INode<'a, K, V>),
}

pub struct Tree23<'a, K, V> {
    root: Option<Node<'a, K, V>>,
}

impl<K: Ord, V> LeafNode<K, V> {
    pub fn get(&self, key: &K) -> Option<&V> {
        if key == &self.key {
            Some(&self.value)
        } else {
            None
        }
    }
}

impl<'a, K, V> Tree23<'a, K, V> {
    pub fn new() -> Tree23<'a, K, V> { Tree23 { root: None } }

    fn leaf(key: K, value: V) -> Tree23<'a, K, V> {
        Tree23 {
            root: Some(Node::Leaf(LeafNode { key, value, })),
        }
    }
}

impl<'a, K: Ord, V> INode<'a, K, V> {
    pub fn get(&self, key: &K) -> Option<&V> {
        use INode::*;
        use std::cmp::Ordering::*;
        match self {
            TwoNode { field, children } => match key.cmp(field) {
                Less => children[0].get(key),
                Equal => children[1].get(key),
                Greater => children[1].get(key),
            },
            ThreeNode { fields, children } => match key.cmp(fields[0]) {
                Less => children[0].get(key),
                Equal => children[1].get(key),
                Greater => match key.cmp(fields[1]) {
                    Less => children[1].get(key),
                    _ => children[2].get(key),
                },
            }
        }
    }
}

impl<'a, K: Ord, V> Node<'a, K, V> {
    fn get(&self, key: &K) -> Option<&V> {
        match self {
            Node::Leaf(ref leaf) => leaf.get(key),
            Node::Interior(ref node) =>  node.get(key),
        }
    }

    fn insert(&mut self, key: K, value: V) {
        use std::cmp::Ordering::*;
        let mut new_self = None;
        match self {
            Node::Leaf(ref mut n) => match key.cmp(&n.key) {
                Equal => n.value = value,
                Less => new_self = {
                    let new_leaf = LeafNode {
                        key: key,
                        value: value,
                    };
                },
                Greater => new_self = todo!(),
            }
            _ => (),
        }
    }
}

impl<'a, K: Ord, V> Tree23<'a, K, V> {
    pub fn get(&self, key: &K) -> Option<&V> {
        match self.root {
            None => None,
            Some(ref n) => n.get(key),
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        match self.root {
            None => *self = Tree23::leaf(key, value),
            Some(ref mut n) => n.insert(key, value),
        }
    }
}

#[test]
fn construct_empty() {
    let tree = Tree23::<usize, String>::new();
}
