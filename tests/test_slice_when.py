from pasio.utils.slice_when import slice_when

def test_slice_when():
    big_diff = lambda x,y: x + 1 < y
    assert [list(g) for g in slice_when([1,2,3,7,8,9],       big_diff)] == [[1,2,3], [7,8,9]]
    assert [list(g) for g in slice_when([1,2,3,7,8,9,99],    big_diff)] == [[1,2,3], [7,8,9], [99]]
    assert [list(g) for g in slice_when([-5,1,2,3,7,8,9],    big_diff)] == [[-5], [1,2,3], [7,8,9]]
    assert [list(g) for g in slice_when([1,2,3],             big_diff)] == [[1,2,3]]
    assert [list(g) for g in slice_when([],                  big_diff)] == []
    assert [list(g) for g in slice_when([1],                 big_diff)] == [[1]]
    assert [list(g) for g in slice_when([1, 1],              big_diff)] == [[1, 1]]
    assert [list(g) for g in slice_when([1, 1],              big_diff)] == [[1, 1]]
    assert [list(g) for g in slice_when([1, 1.5],             big_diff)] == [[1, 1.5]]
    assert [list(g) for g in slice_when([1,1,1, 3,3, 5,5,5], big_diff)] == [[1,1,1], [3,3], [5,5,5]]

    # exhausts a group, when you request next group
    sl = slice_when([1,1,1, 3,3, 5,5,5], big_diff)
    next(sl)
    assert [list(g) for g in sl] == [[3,3], [5,5,5]]

    # exhausts group from which elements were picked, when you request next group
    sl = slice_when([1,1,1, 3,3, 5,5,5], big_diff)
    it = next(sl)
    next(it)
    assert [list(g) for g in sl] == [[3,3], [5,5,5]]

    # works with iterators
    assert [list(g) for g in slice_when(range(5), big_diff)] == [[0,1,2,3,4]]
