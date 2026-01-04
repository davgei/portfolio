class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class Set:
    def __init__(self):
        self.root = None
        self.size_count = 0
    
    def insert(self, key):
        # Rekursiv funksjon for å sette inn et nytt element
        def _insert(node, key):
            if node is None:
                return Node(key)
            if key < node.key:
                node.left = _insert(node.left, key)
            elif key > node.key:
                node.right = _insert(node.right, key)
            return node
        
        if not self.contains(key):  # Sjekk først om nøkkelen finnes
            self.root = _insert(self.root, key)
            self.size_count += 1

    def contains(self, key):
        # Rekursiv funksjon for å sjekke om et element finnes
        def _contains(node, key):
            if node is None:
                return False
            if key == node.key:
                return True
            elif key < node.key:
                return _contains(node.left, key)
            else:
                return _contains(node.right, key)
        
        return _contains(self.root, key)

    def remove(self, key):
        # Rekursiv funksjon for å fjerne et element
        def _remove(node, key):
            if node is None:
                return node
            if key < node.key:
                node.left = _remove(node.left, key)
            elif key > node.key:
                node.right = _remove(node.right, key)
            else:
                if node.left is None:
                    return node.right
                elif node.right is None:
                    return node.left
                temp = _min_value_node(node.right)
                node.key = temp.key
                node.right = _remove(node.right, temp.key)
            return node
        
        # Hjelpefunksjon for å finne noden med minst verdi i et tre
        def _min_value_node(node):
            current = node
            while current.left is not None:
                current = current.left
            return current
        
        if self.contains(key):  # Fjern bare hvis nøkkelen finnes
            self.root = _remove(self.root, key)
            self.size_count -= 1

    def size(self):
        return self.size_count

def main():
    n = int(input())  # Les inn antall operasjoner
    s = Set()

    for _ in range(n):
        command = input().split()
        if command[0] == 'insert':
            x = int(command[1])
            s.insert(x)
        elif command[0] == 'contains':
            x = int(command[1])
            print("true" if s.contains(x) else "false")
        elif command[0] == 'remove':
            x = int(command[1])
            s.remove(x)
        elif command[0] == 'size':
            print(s.size())

if __name__ == "__main__":
    main()