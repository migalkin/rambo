import sys
sys.path.append('..')
import load


def test_gather_entities():
    """ Test DataManager.gather_entities """
    # Make a dataset, dummy
    num_entities = 10
    triples = [
        [2, 14, 3, 23, 7],
        [2, 18, 1, 20, 8],
        [5, 17, 0, 110, 9],
        [4, 42, 6, 123, 7]
    ]
    pos = [0, 2]

    res = load.DataManager.gather_entities(data=triples, n_ents=num_entities, positions=pos)

    # Expect that nothing in gather entities would be more than 6.
    assert (res < 7).any(), 'gather_entities returned an entity which only appeared in "non-corrupting positions"'
    for ent in range(7):
        assert ent in res, f"Entity nr {ent} should have been here!"


def test_gather_entities_b():
    """ Test DataManager.gather_entities, another round """
    # Make a dataset, dummy
    num_entities = 10
    triples = [
        [2, 14, 3, 23, 7],
        [2, 18, 1, 20, 8],
        [5, 17, 7, 110, 9],
        [4, 42, 6, 123, 7]
    ]
    pos = [0, 2]

    res = load.DataManager.gather_entities(data=triples, n_ents=num_entities, positions=pos)

    # Expect that nothing in gather entities would be more than 6.
    assert (res < 8).any(), 'gather_entities returned an entity which only appeared in "non-corrupting positions"'
    assert 0 not in res, "Entity 0 should not be in the returned entity"
    for ent in range(1, 8):
        assert ent in res, f"Entity nr {ent} should have been here!"


if __name__ == "__main__":
    # Run the tests here
    test_gather_entities()