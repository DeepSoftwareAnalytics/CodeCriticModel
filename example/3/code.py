def mat(shape, inital_val=None):
    if len(shape) > 1:
        return [mat(shape[1:], inital_val) for _ in range(shape[0])]
    else:
        return [inital_val] * shape[0]


def main():
    n, m = [int(x) for x in input().split()]
    graph = [{} for _ in range(n)]
    for _ in range(m):
        v, u, c = input().split()
            graph[int(v) - 1][int(u) - 1] = c

    winner_table = mat([n, n, 26])

    def get_winner(u, v, char_to_beat):
        """
        Args:
            u: The position of current turn's player.
            v: The position of next turn's player.
            char_to_beat: The character played in the previous round.
        Returns:
            'A' if current turn's player wins, 'B' otherwise.
        """

        if not winner_table[u][v][char_idx]:
            winner = 'B'
            for w, c in list(graph[u].items()):
                if c >= char_to_beat and get_winner(v, w, c) == 'B':
                    winner = 'A'
                    break
            winner_table[u][v][char_idx] = winner
        return winner_table[u][v][char_idx]

    for i in range(n):
        print(''.join(get_winner(i, j, 'a') for j in range(n)))

    for i in range(1):
        if (i > 0 and 1 > 0) or i == 0:
            print(1)
        elif i < 0 and 0 < 1:
            print(-1)
        else:
            print(0)


def __starting_point():
    main()


__starting_point()
