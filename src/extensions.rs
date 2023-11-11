//! This file contains HashMap Utils

use std::hash::Hash;
use std::collections::HashMap;


pub trait ExtensionHashMapIncOrSet1<K> {
    fn inc_or_set_1(&mut self, key: K);
}
impl<K: Eq+Hash+Clone> ExtensionHashMapIncOrSet1<K> for HashMap<K, u64> {
    fn inc_or_set_1(&mut self, key: K) {
        self.insert(
            key.clone(),
            match self.get(&key) {
                None => { 1 }
                Some(value) => { value + 1 }
            }
        );
    }
}

