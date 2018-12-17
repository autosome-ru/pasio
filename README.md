# PASIO

Pasio is a tool for denosing DNA coverage profiles coming from high-throughput sequencing data.
Example of experiments pasio works well on is ChIP-seq, DNAse-seq, ATAC-seq.

It takes a .bed file of counts (integer values, normalization is not supported). And produces
tsv file with genome splited into segments which coverage can be treated as equal.

Pasio runs on both Python 2 and 3 (Python 2 interpreter runs a bit faster).
The only dependencies are numpy and scipy.

Recommended command line for most practical cases is:

```
python src/pasio.py
      --bedgraph <PATH TO INPUT bedGraph FILE> -o <PATH TO OUTPUT bedGraph FILE>
      --alpha 5 --beta 1
      --no_split_constant
      --algorithm rounds
      --window_shift 1250 --window_size 2500
```

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

In order to split a chromosome one of splitters is invoked.
They return the best segmentation split points and the score of this segmentation.
Some splitters accept `split_candidates` parameter; in that case only splits
at specified positions are allowed.

Splitters:
* The most basic splitter is `SquareSplitter` which implements dynamic programming algorithm
   with `O(N^2)` complexity where `N` is a number of split candidates. Other splitters perform
   some heuristical optimisations on top of `SquareSplitter`
* `SlidingWindowSplitter` tries to segment not an entire contig (chromosome) but shorter parts of contig.
   So they scan a sequence with a sliding window and remove split candidates which are unlikely.
   Each window is processed using `SquareSplitter`. Candidates from different windows are then aggregated.
* `RoundSplitter` perform the same procedure and repeat it for several rounds or until list of split candidates converges.
* `NotZeroSplitter` doesn't try to make splits inside of zero-valued interval.
* `NotConstantSplitter` doesn't try to make splits inside of a constant-valued interval. More precisely,
   it drops split points which don't differ from a left-adjacent point.

Splits denote segment boundaries to the left of position. Adjacent splits `a` and `b` form semi-closed interval `[a, b)`
E.g. for coverage counts `[99,99,99, 1,1,1]` splits should be `[0, 3, 6]`.
So that we have two segments: `[0, 3)` and `[3, 6)`.

It's guaranted that splits and split candidates are stored as numpy arrays and always include the points just before and after the contig.

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
