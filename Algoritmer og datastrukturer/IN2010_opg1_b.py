class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1  # Høyden til en ny node er alltid 1

class AVLSet:
    def __init__(self):
        self.root = None
        self.size_count = 0
    
    def insert(self, key):
        def _insert(node, key):
            if node is None:
                return Node(key)

            # Standard BST innsetting
            if key < node.key:
                node.left = _insert(node.left, key)
            elif key > node.key:
                node.right = _insert(node.right, key)
            else:
                return node  # Nøkkelen finnes allerede, ikke sett inn duplikater

            # Oppdater høyden på denne forfaderen
            node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

            # Få balansefaktoren til denne noden for å sjekke om den er ubalansert
            balance = self._get_balance(node)

            # Venstre-venstre tilfelle (rotasjon høyre)
            if balance > 1 and key < node.left.key:
                return self._right_rotate(node)

            # Høyre-høyre tilfelle (rotasjon venstre)
            if balance < -1 and key > node.right.key:
                return self._left_rotate(node)

            # Venstre-høyre tilfelle (dobbelrotasjon høyre-venstre)
            if balance > 1 and key > node.left.key:
                node.left = self._left_rotate(node.left)
                return self._right_rotate(node)

            # Høyre-venstre tilfelle (dobbelrotasjon venstre-høyre)
            if balance < -1 and key < node.right.key:
                node.right = self._right_rotate(node.right)
                return self._left_rotate(node)

            return node

        if not self.contains(key):
            self.root = _insert(self.root, key)
            self.size_count += 1

    def contains(self, key):
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
                
                temp = self._min_value_node(node.right)
                node.key = temp.key
                node.right = _remove(node.right, temp.key)
            
            if node is None:
                return node

            node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

            balance = self._get_balance(node)

            # Venstre-venstre tilfelle
            if balance > 1 and self._get_balance(node.left) >= 0:
                return self._right_rotate(node)

            # Venstre-høyre tilfelle
            if balance > 1 and self._get_balance(node.left) < 0:
                node.left = self._left_rotate(node.left)
                return self._right_rotate(node)

            # Høyre-høyre tilfelle
            if balance < -1 and self._get_balance(node.right) <= 0:
                return self._left_rotate(node)

            # Høyre-venstre tilfelle
            if balance < -1 and self._get_balance(node.right) > 0:
                node.right = self._right_rotate(node.right)
                return self._left_rotate(node)

            return node
        
        if self.contains(key):
            self.root = _remove(self.root, key)
            self.size_count -= 1

    def size(self):
        return self.size_count

    def _get_height(self, node):
        if node is None:
            return 0
        return node.height

    def _get_balance(self, node):
        if node is None:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _left_rotate(self, z):
        y = z.right
        T2 = y.left

        # Utfør rotasjon
        y.left = z
        z.right = T2

        # Oppdater høyder
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

        # Returner ny rot
        return y

    def _right_rotate(self, z):
        y = z.left
        T3 = y.right

        # Utfør rotasjon
        y.right = z
        z.left = T3

        # Oppdater høyder
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

        # Returner ny rot
        return y

    def _min_value_node(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current


def main():
    n = int(input())  # Les inn antall operasjoner
    s = AVLSet()

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