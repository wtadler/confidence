[![DOI](https://zenodo.org/badge/49356719.svg)](https://zenodo.org/badge/latestdoi/49356719)


# Are human confidence reports Bayesian?
This repository accompanies three manuscripts on the topic of whether human confidence reports are Bayesian. The following manuscripts, by [Will Adler](http://wtadler.com), [Rachel Denison](http://racheldenison.com), [Marisa Carrasco](https://psych.nyu.edu/carrasco/), and [Wei Ji Ma](http://www.cns.nyu.edu/malab/), are in submission with preprints available on bioRxiv:

##### 1. Confidence when uncertainty is due to stimulus reliability
William T. Adler, Wei Ji Ma. (2018, in press; 2016, posted on *bioRxiv*). [Comparing Bayesian and non-Bayesian accounts of human confidence reports](https://www.biorxiv.org/content/early/2018/01/29/093203). *PLoS Computational Biology.*

##### 2. Confidence when uncertainty is due to inattention
Rachel N. Denison*, William T. Adler*, Marisa Carrasco, Wei Ji Ma. (2018). [Humans incorporate attention-dependent uncertainty into perceptual decisions and confidence](http://www.pnas.org/content/early/2018/10/05/1717720115) [[pdf]](http://www.wtadler.com/papers/Denison2018.pdf). *Proceedings of the National Academies of Sciences.*

##### 3. Theoretical exploration of Bayesian confidence signatures.
William T. Adler, Wei Ji Ma. (2018). [Limitations of proposed signatures of Bayesian confidence](https://www.mitpressjournals.org/doi/abs/10.1162/neco_a_01141?journalCode=neco) [[pdf]](http://www.wtadler.com/papers/Adler2018.pdf). *Neural Computation.*


# Usage

## Papers 1 and 2

The below instructions apply to Papers 1 and 2. The [code](signatures) for Paper 3 is relatively straightforward and you can find a different README with the corresponding code.

### Setup

Clone or download this repository. In MATLAB, `cd` into this repository. Use `addpath(genpath('helper_functions'))` to add the necessary functions to your path.

### Plot some data

Use `show_data_or_fits` to plot the human data. Try this to plot data from Paper 1:
```
axis.col = 'slice';
axis.row = 'depvar';
axis.fig = 'task';

show_data_or_fits('root_datadir', 'human_data/reliability_exp1', ...
                  'axis', axis, ...
                  'depvars', {'resp', 'Chat', 'tf'}, ...
                  'tasks', {'A', 'B'}, ...
                  'slices', {'c_s', 'c', 'c_C'});
```
To get two figures. The second one, for Task B, should look like this:
![](http://wtadler.com/picdrop/reliability_fig.png)

The parameters in `axis` indicate that:
- The columns of the resulting figures should show different slices of the data. In this case, the data in the first column slices the data by reliability _c_ and orientation _s_ (slice `c_s`); the second column slices the data by just reliability _c_ (slice `c`); and the third column slices the data by reliability _c_ and true category _C_ (slice `c_C`).
- The rows of the resulting figures should show different dependent variables on the y-axis. In this case, those variables are mean button press (`resp`), proportional response "category 1" (`Chat`, as in C-hat, used in the papers), and proportion correct (`tf`, as in true-false).
- You will get a figure like this for each task. In this case, a dataset was chosen that includes two tasks, and both tasks `A` and `B` were specified as arguments, so you will get two figures.

Options for the axis parameters are `'none'`, `'slice'`, `'depvar'`, `'task'`, `'subject'`, and `'model'`. `'subject'` will give you individual subject data rather than grouped, as shown above. ``'model'`` will show you fits to different models (more on that below).

Options for `depvars` are:
- `resp` Mean button press (choice and confidence), between 1 and 8
- `Chat` Proportion response "category 1"
- `tf` Proportion correct
- `g` Mean confidence, between 1 and 4
- `rt` Reaction time (s)

Options for `slices` are `'s'`, `'Chat'`, `'g'`, `'resp'`, `'rt'`, `'C_s'`, `'c'`, `'c_s'`, `'c_C'`, `'c_Chat'`, `'c_g'`, and `'c_resp'`, which are just various combinations of the variables described above.

To plot data from Paper 2, try running this code:
```
axis.col = 'slice';
axis.row = 'depvar';
axis.fig = 'none';

show_data_or_fits('root_datadir', 'human_data/attention', ...
                  'axis', axis, ...
                  'depvars', {'resp', 'Chat', 'tf'}, ...
                  'slices', {'c_s', 'c', 'c_C'});
```
to get something like this:
![](http://wtadler.com/picdrop/attention_fig.png)

### Explore model fits

If you want to explore model fits, try going to this [repository](https://zenodo.org/record/1458240) and downloading the .mat files into the [model_fits](model_fits) folder. Be warned that these are 11 files averaging about 600MB. They're big because they include the thinned chains of samples generated when we fit the models with MCMC.

Once downloaded, run [`load_model_fits.m`](load_model_fits.m) to load in all the files and organize the models as they are used in Paper 1. Now you will have several loaded structs containing all you need to know about model parameters (e.g., lower and upper bounds) and fits (e.g., MCMC samples and scores to each subject). For instance, the `attention.modelmaster` struct will contain all the models used in Paper 2. Try loading and printing out the names of the models:
```
load_model_fits
models = attention.modelmaster;
rename_models(models);
>>> 1: Bayes_{Strong}-{\itd}N + non-param. \sigma, B
>>> 2: Bayes_{Weak}-{\itd}N + non-param. \sigma, B
>>> 3: Orientation Estimation + non-param. \sigma, B
>>> 4: Linear Neural + non-param. \sigma, B
>>> 5: Lin + non-param. \sigma, B
>>> 6: Quad + non-param. \sigma, B
>>> 7: Fixed + non-param. \sigma, B
>>> 8: Bayes-{\itd}N + non-param. \sigma, B, choice only
>>> 9: Orientation Estimation + non-param. \sigma, B, choice only
>>> 10: Linear Neural + non-param. \sigma, B, choice only
>>> 11: Lin + non-param. \sigma, B, choice only
>>> 12: Quad + non-param. \sigma, B, choice only
>>> 13: Fixed + non-param. \sigma, B, choice only
```

If you're interested in the Quad confidence model (model #6), for instance, here's some of the information you could pull out of the struct:
- `models(6).parameter_names`: names of the model parameters
- `models(6).lb`: parameter lower bound values
- `models(6).ub`: parameter upper bound values
- `samples = vertcat(models(6).extracted(1).p{:})`: MCMC samples (nSamples x nParams) for the model, subject 1. (Each cell in `models(12).extracted(1).p` is an MCMC chain)

With the models loaded, you can also use `show_data_or_fits` to plot model fits and scores. Try running this code:

```
axis.col = 'model';
axis.row = 'depvar';
axis.fig = 'none';

show_data_or_fits('root_datadir', 'human_data/attention', ...
                  'axis', axis, ...
                  'depvars', {'resp', 'tf'}, ...
                  'slices', {'c_s', 'c', 'c_C'}, ...
                  'models', models([1:2 5:7]), ...
                  'MCM', 'loopsis');
```
to get a plot that includes model fits as well as [PSIS-LOO](https://arxiv.org/abs/1507.04544) score comparisons for five models:
![](http://wtadler.com/picdrop/attention_fits_fig.png)



### Understanding the computations behind each model
There are two functions that you might want to look through if you are interested in how the computations described in the paper are implemented in the code. One function is [`trial_generator.m`](helper_functions/trial_generator.m), which shows how fake data is generated for each model. The other one is [`nloglik_fcn.m`](helper_functions/nloglik_fcn.m) which is used for computing the (negative) log-likelihood of each model.

### Run the psychophysical experiment
To run the experiment, you'll need to install [Psychtoolbox](http://psychtoolbox.org/). Then [`run_categorical_decision.m`](run_categorical_decision.m) is used to set the parameters of the experiment and run it.


## Paper 3

All of the code necessary to produce the simulations in Paper 3, as well as a README specific to Paper 3, can be found in the [signatures](signatures) folder. Paper 3 involves no human data.
