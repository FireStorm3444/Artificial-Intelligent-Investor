class TrieNode:
    def __init__(self):
        self.children = {}
        self.stocks = set()

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, stock_object):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.stocks.add(stock_object)

    def __dfs(self, node, prefix, results):
        if node.stocks:
            results.extend([stock for stock in node.stocks])
        # Traverse all children nodes
        for char, child_node in node.children.items():
            self.__dfs(child_node, prefix + char, results)

    def search_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        results = []
        self.__dfs(node, prefix, results)
        return results