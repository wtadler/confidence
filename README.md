# Are human confidence reports Bayesian?
This repository accompanies three manuscripts on the topic of whether human confidence reports are Bayesian. The following manuscripts, by [Will Adler](http://wtadler.com), [Rachel Denison](http://racheldenison.com), [Marisa Carrasco](https://psych.nyu.edu/carrasco/), and [Wei Ji Ma](http://www.cns.nyu.edu/malab/), are in submission with preprints available on bioRxiv:

##### 1. Confidence when uncertainty is due to stimulus reliability
William T. Adler, Wei Ji Ma. (2016). [Comparing Bayesian and non-Bayesian accounts of human confidence reports](https://www.biorxiv.org/content/early/2018/01/29/093203).

##### 2. Confidence when uncertainty is due to inattention
Rachel N. Denison, William T. Adler, Marisa Carrasco, Wei Ji Ma. (2017). [Humans flexibly incorporate attention-dependent uncertainty into perceptual decisions and confidence](https://www.biorxiv.org/content/early/2017/08/10/175075).

##### 3. Theoretical exploration of Bayesian confidence signatures.
William T. Adler, Wei Ji Ma. (2017). [Limitations of proposed signatures of Bayesian confidence](https://www.biorxiv.org/content/early/2018/01/29/218222).


# Usage

## Papers 1 and 2

The below instructions apply to papers 1 and 2. The [code](signatures) for paper 3 is relatively straightforward and you can find a different README with the corresponding code.

### Setup

Clone or download this repository. In MATLAB, `cd` into this repository. Use `addpath(genpath('helper_functions'))` to add the necessary functions to your path.

### Plot some data

Use `show_data_and_fits` to plot the human data. Try this to plot data from paper 1:
```
axis.col = 'slice'
axis.row = 'depvar'
axis.fig = 'task'

show_data_or_fits('root_datadir', 'human_data/reliability_exp1', ...
                  'axis', axis, ...
                  'depvars', {'resp', 'Chat', 'tf'}, ...
                  'tasks', {'A', 'B'}
                  'slices', {'c_s', 'c', 'c_C'})
```
To get two figures. The second one, for Task B, should look like this:
![](http://wtadler.com/picdrop/reliability_fig.png)

To plot data from paper 2, try running the same code as above except replace `'human_data/reliability_exp1'` with `'human_data/attention'`. You'll get something like:
![](http://wtadler.com/picdrop/attention_fig.png)

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


### Explore model fits

If you want to explore model fits, try going to this  [Google Drive folder](https://drive.google.com/drive/folders/13PCbl8IQg7tsL49F1o-t0RuI-818BXTb?usp=sharing) and downloading the .mat files into the [model_fits](model_fits) folder. Be warned that these are 11 files averaging about 600MB. They're big because I am saving the chains of samples generated when we fit the models with MCMC.

Once downloaded, run [`load_model_fits.m`](load_model_fits.m) to load in all the files and organize the models as they are used in Paper 1.

_More info to be written about how to explore model types, fits, and parameters, and use the aforementioned `show_data_and_fits` to plot model fit scores, fits to the data._

### Understanding the computations behind each model
There are two functions that you might want to look through if you are interested in the computations behind each model. One function is [`trial_generator.m`](helper_functions/trial_generator.m), which shows how fake data is generated for each model. The other one is [`nloglik_fcn.m`](helper_functions/nloglik_fcn.m) which is used for computing the (negative) log-likelihood of each model.

### Run the psychophysical experiment
To run the experiment, you'll need to install [Psychtoolbox](http://psychtoolbox.org/). Then [`run_categorical_decision.m`](run_categorical_decision.m) is used to set the parameters of the experiment and run it.


## Paper 3

All of the code necessary to produce the simulations in paper 3, as well as a README specific to paper 3, can be found in the [signatures](signatures) folder. Paper 3 involves no human data.
