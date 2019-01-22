This folder contains `.mat` files for each session from each subject, from four experiments.

The attention experiment is described in [Humans incorporate attention-dependent uncertainty into perceptual decisions and confidence](http://www.pnas.org/content/early/2018/10/05/1717720115) [[pdf]](http://www.wtadler.com/papers/Denison2018.pdf).

The three reliability experiments are described in [Comparing Bayesian and non-Bayesian accounts of human confidence reports](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006572) [[pdf]](http://www.wtadler.com/papers/Adler2018b.pdf). The file [`reliability_exp1.csv`](reliability_exp1.csv) contains all data for Experiment 1 in this paper. The important columns in this spreadsheet are:

- `subject_name`
- `task`
  - A: Task in which categories have different means, same SDs
  - B: Task in which categories have same means, different SDs
- `block_type`
  - Training: Category training (high stimulus reliability, correctness feedback)
  - Test: Test block (variable stimulus reliability, no trial-to-trial correctness feedback)
- `stim_category` Binary category (-1 or 1) specifying a stimulus distribution
- `stim_type`
  - grate: Drifting oriented Gabor
  - ellipse: Oriented ellipse
- `stim_reliability` Stimulus contrast (for `stim_type` Gabor) or eccentricity (for `stim_type` ellipse)
- `stim_orientation` Stimulus orientation (degrees)
- `resp_buttonid` Subject response (integers -4 through 4, except 0, with -4 indicating "high confidence cat. -1" and 4 indicating "high confidence cat. 1")
- `resp_category` Category choice (-1 or 1)
- `resp_confidence` Confidence report (1 through 4)
- `resp_rt` Reaction time (seconds)
