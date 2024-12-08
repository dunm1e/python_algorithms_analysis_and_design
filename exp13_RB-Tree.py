class Node:
    def __init__(self, color, key, val, left=None, right=None, parent=None):
        self.color = color
        self.key = key
        self.val = val
        self.left = left
        self.right = right
        self.parent = parent


RED = True
BLACK = False


class RedBlackTree:
    def __init__(self):
        self.root = None

    def parent_of(self, node):
        return node.parent if node else None

    def is_red(self, node):
        return node.color == RED if node else False

    def set_color(self, node, color):
        if node:
            node.color = color

    def rotate_left(self, node):
        y = node.right
        if y is None:
            return

        beta = y.left
        node.right = beta
        if beta is not None:
            beta.parent = node

        y.parent = node.parent
        if node.parent is None:
            self.root = y
        elif node == node.parent.left:
            node.parent.left = y
        else:
            node.parent.right = y

        y.left = node
        node.parent = y

    def rotate_right(self, node):
        y = node.left
        if y is None:
            return

        beta = y.right
        node.left = beta
        if beta is not None:
            beta.parent = node

        y.parent = node.parent
        if node.parent is None:
            self.root = y
        elif node == node.parent.right:
            node.parent.right = y
        else:
            node.parent.left = y

        y.right = node
        node.parent = y

    def flip_colors(self, node):
        node.color = RED
        node.left.color = BLACK
        node.right.color = BLACK

    def put(self, key, val):
        if key is None:
            raise ValueError("first argument to put() is null")
        if val is None:
            self.delete(key)
            return
        self.root = self._put(self.root, key, val)
        self.set_color(self.root, BLACK)

    def _put(self, h, key, val):
        if h is None:
            return Node(RED, key, val)

        cur, p = h, None
        while cur:
            p = cur
            compareRes = (key > cur.key) - (key < cur.key)
            if compareRes < 0:
                cur = cur.left
            elif compareRes > 0:
                cur = cur.right
            else:
                cur.val = val
                return h

        newNode = Node(RED, key, val, parent=p)
        if p:
            compareRes = (newNode.key > p.key) - (newNode.key < p.key)
            if compareRes < 0:
                p.left = newNode
            else:
                p.right = newNode
        else:
            self.root = newNode

        self.fix_after_insertion(newNode)
        return self.root

    def fix_after_insertion(self, k):
        while k != self.root and self.is_red(self.parent_of(k)):
            p = self.parent_of(k)
            g = self.parent_of(p)
            if p == g.left:
                u = g.right
                if self.is_red(u):
                    self.flip_colors(g)
                    k = g
                else:
                    if k == p.right:
                        k = p
                        self.rotate_left(p)
                    self.set_color(p, BLACK)
                    self.set_color(g, RED)
                    self.rotate_right(g)
            else:
                u = g.left
                if self.is_red(u):
                    self.flip_colors(g)
                    k = g
                else:
                    if k == p.left:
                        k = p
                        self.rotate_right(p)
                    self.set_color(p, BLACK)
                    self.set_color(g, RED)
                    self.rotate_left(g)
        self.set_color(self.root, BLACK)

    def delete(self, key):
        pass


def print_tree(node, indent="", last=True):
    if node is not None:
        print(indent, "`- " if last else "|- ", node.key, "(RED)" if node.color == RED else "(BLACK)", sep="")
        indent += "   " if last else "|  "
        print_tree(node.left, indent, False)
        print_tree(node.right, indent, True)


def main():
    tree = RedBlackTree()
    # 插入一些键值对
    keys = [10, 20, 30, 15, 25, 5, 1]
    for key in keys:
        tree.put(key, str(key))
        print(f"\nAfter inserting {key}:")
        print_tree(tree.root)

    # 检查根节点颜色是否为黑色
    assert tree.root.color == BLACK, "Root should be black"


if __name__ == "__main__":
    main()
