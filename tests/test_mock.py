import KAS

from KAS import Explorer, Next, MockNodeMetadata, MockNode, MockSampler


def test_mock_sampler():
    # a has index 1, b has index 2, anonymous vertex has index 3
    vertices = ['root', 'a', 'b', {}, {'name': 'final', 'is_final': True}]

    edges = [
        # you can directly use Next or a tuple
        ('root', [(Next(Next.Share, 0), 'a'), ((Next.Share, 1), 'b')]),
        # you can also use index to represent a node
        (1, [('Merge(0)', 3), ('Share(2)', 'final')]),
        ('b', [('Share(3)', 'final')]),
    ]

    sampler = MockSampler(vertices, edges)
    root = sampler.root()
    assert root.mock_get_id() == 0
    a = root.get_child('Share(0)')
    b = root.get_child('Share(1)')
    anon = a.get_child('Merge(0)')
    assert anon is not None
    assert anon.is_dead_end()
    final = b.get_child('Share(3)')
    assert final.is_final()


if __name__ == '__main__':
    test_mock_sampler()
