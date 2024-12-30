![Pasio logo](logos/pasio_256.png?raw=true "Logo")
# PASIO

Pasio is a tool for denosing DNA coverage profiles coming from high-throughput sequencing data.
Example of experiments pasio works well on is ChIP-seq, DNAse-seq, ATAC-seq.

It takes a .bed file of counts (integer values, normalization is not supported). And produces
tsv file with genome splited into segments which coverage can be treated as equal.

Pasio runs on both Python 2 and 3 (Python 2 interpreter runs a bit faster).
The only dependencies are numpy and scipy.

Defaults are reasonable for fast yet almost precise computation, so usually it is enough to run:

```
pasio input.bedgraph
```

This defaults are yet subject to change, so if you want results to be reproducible between versions, please specify all substantial parameters (especially α and β) explicitly!

Note that PASIO process bedgraph contig by contig. Thus bedgraph should be sorted by contig/chromosome!

PASIO can read and write to gzipped files (filename should have `.gz` extension).

PASIO can also process bedgraph from stdin by supplying `-` instead of filename.
It can be useful for transcriptomic data where contigs are short enough to be processed on the fly.


## Installation
PASIO works with Python 2.7.1+ and Python 3.4+. The tool is available on PyPA, so you can install it using pip:

```
  python -m pip install pasio
```


Note that pip install wrapper to run pasio without specifying python. One can use one of two options to run it:

```
pasio <options>...
python -m pasio <options>...
```

The latter option can be useful if you want to run it using a certain python version.

## Underlying math

PASIO is a program to segment chromatin accessibility profile. It accepts a bedgraph file
with coverage of position by DNase cuts (e.g. by 5'-ends of DNase-seq)
and splits each contig/chromosome into segments with different accessibilites in an optimal way.

Method is based on two assumptions:
* cuts are introduced by Poisson process `P(λ)` with `λ` depending on segment
* `λ` are distributed as `λ ~ Г(α, β) = β^α * λ^{α - 1} * exp(-βλ) / Г(α)`

`α` and `β` are the only parameters of segmentation.

Then we can derive (logarithmic) marginal likelyhood `logML` to be optimized.
`logML` for a single segment `S` of length `L` with coverages `(S_1, S_2, ...S_L)` and total coverage `C = \sum_i(S_i)` will be:
`logML(S,α,β) = α*log(β) − log(Γ(α)) + log(Γ(C + α)) − (C + α) * log(L + β) − \sum_i log (S_i!)`

Here `α*log(β) − log(Γ(α))` can be treated as a penalty for segment creation (and is approximately proportional to `α*log(β/α))`.
Total `logML(α,β)` is a sum of logarithmic marginal likelihoods for all segments: `logML(α,β) = \sum_S logML(S,α,β)`.
Given a chromosome coverage profile, term `\sum_S {\sum_i log (S_i!)}` doesn't depend on segmentation.
Value `\sum_S {log(Γ(C + α)) − (C + α) * log(L + β)}` is refered as a `self score` of a segment.
We optimize only segmentation-dependent part of `logML` which is termed just `score`.
This score is a sum of self score of a segment and a penalty for segment creation.

## Program design

`split_bedgraph` loads bedgraph file chromosome by chromosome, splits them into segments and writes into output tsv file.
Coverage counts are stored internally with 1-nt resolution.

Splitting is proceeded in two steps: (a) reduce a list of candidate split points (sometimes this step is omitted),
(b) choose splits from a list of candidates and calculate score of segmentation.
The first step is performed with one of so called *reducers*. The second step is performed
with one of *splitters* (each splitter also implements a reducer interface but not vice versa).

Splitters and reducers:
* The most basic splitter is `SquareSplitter` which implements dynamic programming algorithm
   with `O(N^2)` complexity where `N` is a number of split candidates. Other splitters/reducers perform
   some heuristical optimisations on top of `SquareSplitter`
* `SlidingWindowReducer` tries to segment not an entire contig (chromosome) but shorter parts of contig.
   So they scan a sequence with a sliding window and remove split candidates which are unlikely.
   Each window is processed using some base splitter (typically `SquareSplitter`).
   Candidates from different windows are then aggregated.
* `RoundReducer` perform the same procedure and repeat it for several rounds or until list of split candidates converges.
* `NotZeroReducer` discards (all) splits if all points of an interval under consideration are zeros.
* `NotConstantReducer` discards splits between same-valued points.
* `ReducerCombiner` accept a list of reducers to be sequentially applied. The last reducer can also be a splitter.
In that case combiner allows for splitting and scoring a segmentation. To transform any reducer into splitter one can combine
that reducer with `NopSplitter` - so that split candidates obtained by reducer will be treated as
final splitting and NopSplitter make it possible to calculate its score.

Splits denote segment boundaries to the left of position. Adjacent splits `a` and `b` form semi-closed interval `[a, b)`
E.g. for coverage counts `[99,99,99, 1,1,1]` splits should be `[0, 3, 6]`.
So that we have two segments: `[0, 3)` and `[3, 6)`.

Splits and split candidates are stored as numpy arrays and always include both inner split points and segment boundaries, i.e. point just before config start and right after the contig end.

One can also treat splits as positions between-elements (like in python slices)
```
counts:            |  99   99   99  |   1    1     1  |
splits candidates: 0     1    2     3     4     5     6
splits:            0                3                 6
```
Splitters invoke `LogMarginalLikelyhoodComputer` which can compute `logML` for a splitting (and for each segment).
`LogMarginalLikelyhoodComputer` store cumulative sums of coverage counts at split candidates
and also distances between candidates. It allows one to efficiently compute `logML` and doesn't need
to recalculate total segment coverages each time.

In order to efficiently compute `log(x)` and `log(Г(x))` we precompute values for some first million of integer numbers `x`.
Computation efficiency restricts us to integer values of `α` and `β`. Segment lengths are naturally integer,
coverage counts (and total segment counts) are also integer because they represent numbers of cuts.
`LogComputer` and `LogGammaComputer` store precomputed values and know how to calculate these values efficiently.

## See also
Predecessor of our approach — “[Segmentation of long genomic sequences into domains with homogeneous composition with BASIO software](https://doi.org/10.1093/bioinformatics/17.11.1065)”.

## Development

Bumping new version:
```
VERSION='1.2.3'
echo "__version__ = '${VERSION}'"  >  src/pasio/version.py
git commit -am "bump version ${VERSION}"
git tag "${VERSION}"
git push
git push --tags
rm dist/ build/ -r
python3 setup.py sdist
python3 setup.py bdist_wheel --universal
twine upload dist/*
```
