import KAS

from KAS import Next, MockNodeMetadata, MockNode, MockSampler


def test_mock_sampler():
    # a has index 1, b has index 2, anonymous vertex has index 3
    # you can set arbitrary attributes
    vertices = ['root', 'a', 'b', {'an_interesting_attribute': 'amazing'}, {'name': 'final', 'is_final': True, 'accuracy': 0.5, 'arbitrary_metadata': 42}]

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
    assert anon.mock_get('an_interesting_attribute') == 'amazing'
    final = b.get_child('Share(3)')
    assert final.is_final()
    assert final.mock_get_accuracy() == 0.5
    assert final.mock_get('arbitrary_metadata') == 42


if __name__ == '__main__':
    test_mock_sampler()
