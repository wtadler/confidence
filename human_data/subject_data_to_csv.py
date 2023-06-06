import numpy as np
import os
import datetime
import re
import pandas as pd
import scipy.io as spio

### https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

###

class Vividict(dict):
    def __missing__(self, key):
        value = self[key] = type(self)() # retain local pointer to value
        return value        
        


def oldcat2newcat(x):
    # turns 1 and 2 into -1 and 1
    return 2 * np.array(x,np.int8) - 3

#%%
# 1. run block with these settings:
# experiments = Vividict({
#                'reliability_exp1': {
#                    'A':'reliability_exp1/taskA',
#                    'B':'reliability_exp1/taskB'}})
# block_types = ['Test','Training']


# 2. and then with these:
block_types = ['Test']
experiments = Vividict({
               'reliability_exp2': {
                   'A':'reliability_exp2/taskA',
                   'B':'reliability_exp2/taskB'},
               # 'reliability_exp3': {
               #     'B':'reliability_exp3'}
               })


for experiment in experiments:
    experiments[experiment]['subjects'] = Vividict()
    for task, taskdir in [i for i in experiments[experiment].items() if i[0]!='subjects']:
        for file in os.listdir(taskdir):
            if file.endswith('.mat'):
                name, date, time = re.split('_',file)[0:3]
                if len(time)==8: # some have older time format
                    time = time[:4]+'00'+'.mat'
                
                if name not in experiments[experiment]['subjects']:
                    experiments[experiment]['subjects'][name]['df'] = pd.DataFrame()

                session_start_time = datetime.datetime(int(date[0:4]),int(date[4:6]),int(date[6:8]),int(time[0:2]),int(time[2:4]),min(59,int(time[4:6])))
                
                mat = loadmat(taskdir+'/'+file)
                stim_type = mat['P']['stim_type']

                for block_type in block_types:
                    tmpdata = mat[block_type]
                    stim_dict = tmpdata['R']
                    resp_dict = tmpdata['responses']
                    
                    if isinstance(resp_dict, dict):
                        resp_dict = [resp_dict]
                    
                    n_blocks = len(resp_dict) # len(stim_dict['draws'])
                    
                    for block_no in range(0, n_blocks):
                        # for blocks with only one section, wrap in a list to make the indexing work better
                        if np.array(stim_dict['draws'][block_no]).ndim==1:
                            for d in ['draws', 'sigma', 'phase', 'trial_order']:
                                stim_dict[d][block_no]=[stim_dict[d][block_no]]
                            for d in ['tf', 'c', 'conf', 'rt']:
                                resp_dict[block_no][d]=[resp_dict[block_no][d]]
                        
                        
                        n_sections = len(tmpdata['R']['draws'])
                        for section_no in range(0, n_sections):
                            stim_category = oldcat2newcat(stim_dict['trial_order'][block_no][section_no])
                            resp_category = oldcat2newcat(resp_dict[block_no]['c'][section_no])

                            tmpdf = pd.DataFrame({'stim_orientation': stim_dict['draws'][block_no][section_no],
                                                  'stim_reliability': stim_dict['sigma'][block_no][section_no],
                                                  'stim_phase': stim_dict['phase'][block_no][section_no],
                                                  'stim_category': stim_category,
                                                  'stim_type': stim_type,
                                                  'resp_confidence': resp_dict[block_no]['conf'][section_no],
                                                  'resp_category': resp_category,
                                                  'resp_buttonid': resp_category*resp_dict[block_no]['conf'][section_no],
                                                  'resp_correct': resp_dict[block_no]['tf'][section_no],
                                                  'resp_rt': resp_dict[block_no]['rt'][section_no],
                                                  'resp_rtConf': resp_dict[block_no]['rtConf'][section_no], # uncomment this for experiment 2
                                                  'block_no': block_no,
                                                  'section_no': section_no,
                                                  'task': task,
                                                  'block_type': block_type,
                                                  'session_start_time': session_start_time,
                                                  })
                            experiments[experiment]['subjects'][name]['df'] = pd.concat([experiments[experiment]['subjects'][name]['df'], tmpdf], ignore_index = True)
                            experiments[experiment]['subjects'][name]['df']['subject_name'] = name


for experiment in experiments:
    dfs = [i[1]['df'] for i in experiments[experiment]['subjects'].items()]
    sort_cols = ['subject_name', 'session_start_time', 'task', 'block_type', 'block_no', 'section_no']
    ascending = [False if i=='block_type' else True for i in sort_cols]

    cols = ['subject_name', 'session_start_time', 'task', 'block_type', 'block_no', 'section_no', 'stim_type', 'stim_category', 'stim_reliability', 'stim_orientation', 'stim_phase', 'resp_buttonid', 'resp_category', 'resp_confidence', 'resp_rt', 'resp_correct']#, 'resp_rtConf'] # uncomment for exp 2

    all = pd.concat(dfs, ignore_index=True)
    all = all.sort_values(sort_cols, ascending=ascending)[cols]
    all.to_csv(f'{experiment}.csv', index=False)

#%%

# 3. generate this .mat file in matlab using compile_and_analyze_data() and save(), and then run:
att = loadmat('attention_data.mat')['real_data']['B']['data']

df = pd.DataFrame()
for subj in att:
    raw = subj['raw']
    tmpdf = pd.DataFrame({'subject_name': subj['name'],
                          'task': 'B',
                          'block_type': 'Test',
                          'stim_type': 'grate',
                          'stim_category': raw['C'],
                          'stim_orientation': raw['s'],
                          'stim_cue_validity': raw['cue_validity'],
                          'stim_cue_validity_id': raw['cue_validity_id'],
                          'resp_confidence': raw['g'],
                          'resp_correct': raw['tf'],
                          'resp_category': raw['Chat'],
                          'resp_rt': raw['rt']})
    df = pd.concat([df, tmpdf], ignore_index=True)

df.to_csv('attention.csv', index=False)


#%%
# run this to clean up for doby's database

def newcat2oldcat(x):
    # turns -1 and 1 into 1 and 2
    return (3 + np.array(x, np.int8)) / 2


df = pd.read_csv('/Users/wtadler/Repos/confidence/human_data/reliability_exp1.csv')
df['subject_name'] = pd.Categorical(df['subject_name'])
df['subject_name'] = df['subject_name'].cat.codes+1

df['stim_category'] = df['stim_category'].apply(newcat2oldcat).astype(int)
df['resp_category'] = df['resp_category'].apply(newcat2oldcat).astype(int)

df = df[df['block_type']=='Test']

ellipse = df[df['stim_type']=='ellipse']
grate = df[df['stim_type']=='grate']

grate['Difficulty'] = pd.Categorical(grate['stim_reliability'])
grate['Difficulty'] = grate['Difficulty'].cat.codes+1

ellipse['Difficulty'] = pd.Categorical(ellipse['stim_reliability'])
ellipse['Difficulty'] = ellipse['Difficulty'].cat.codes+1

df = pd.concat([grate, ellipse])
df['Difficulty'] = 7 - df['Difficulty']

df['Difficulty'].value_counts()
df[df['Difficulty']==1]['resp_correct'].mean()


df = df.rename(columns={'subject_name': 'Subj_idx',
                   'stim_category': 'Stimulus',
                   'stim_orientation': 'Orientation',
                   'resp_category': 'Response',
                   'resp_rt': 'RT_decConf',
                   'resp_confidence': 'Confidence',
                   'task': 'Task'
                   })

cols = ['Subj_idx', 'Stimulus', 'Response', 'Confidence', 'RT_decConf', 'Difficulty', 'Orientation', 'Task']
df[cols].sort_values('Subj_idx').to_csv('for_rahnev_database/reliability_exp1_doby.csv', index=False)



df = pd.read_csv('/Users/wtadler/Repos/confidence/human_data/reliability_exp2.csv')

df['subject_name'] = pd.Categorical(df['subject_name'])
df['subject_name'] = df['subject_name'].cat.codes+1

df['stim_category'] = df['stim_category'].apply(newcat2oldcat).astype(int)
df['resp_category'] = df['resp_category'].apply(newcat2oldcat).astype(int)

df = df[df['block_type']=='Test']

df['Difficulty'] = pd.Categorical(df['stim_reliability'])
df['Difficulty'] = df['Difficulty'].cat.codes+1
df['Difficulty'] = 7 - df['Difficulty']
df['Difficulty'].value_counts()
df[df['Difficulty']==1]['resp_correct'].mean()

df = df.rename(columns={'subject_name': 'Subj_idx',
                   'stim_category': 'Stimulus',
                   'stim_orientation': 'Orientation',
                   'resp_category': 'Response',
                   'resp_rt': 'RT_dec',
                   'resp_confidence': 'Confidence',
                   'resp_rtConf': 'RT_conf',
                   'task': 'Task'})

cols = ['Subj_idx', 'Stimulus', 'Response', 'Confidence', 'RT_dec', 'RT_conf', 'Difficulty', 'Orientation', 'Task']

df[cols].to_csv('/Users/wtadler/Repos/confidence/human_data/for_rahnev_database/reliability_exp2_doby.csv', index=False)


df.groupby('subject_name').count()


df = pd.read_csv('/Users/wtadler/Repos/confidence/human_data/reliability_exp3.csv')
df['subject_name'] = pd.Categorical(df['subject_name'])
df['subject_name'] = df['subject_name'].cat.codes+1

df['stim_category'] = df['stim_category'].apply(newcat2oldcat).astype(int)
df['resp_category'] = df['resp_category'].apply(newcat2oldcat).astype(int)

df = df[df['block_type']=='Test']

df['Difficulty'] = pd.Categorical(df['stim_reliability'])
df['Difficulty'] = df['Difficulty'].cat.codes+1
df['Difficulty'] = 7 - df['Difficulty']

df = df.rename(columns={'subject_name': 'Subj_idx',
                   'stim_category': 'Stimulus',
                   'stim_orientation': 'Orientation',
                   'resp_category': 'Response',
                   'resp_rt': 'RT_decConf',
                   'resp_confidence': 'Confidence'
                   })



cols = ['Subj_idx', 'Stimulus', 'Response', 'Confidence', 'RT_decConf', 'Difficulty', 'Orientation']
df[cols].sort_values('Subj_idx').to_csv('for_rahnev_database/reliability_exp3_doby.csv', index=False)




df = pd.read_csv('/Users/wtadler/Repos/confidence/human_data/attention.csv')

df['subject_name'] = pd.Categorical(df['subject_name'])
df['subject_name'] = df['subject_name'].cat.codes+1

df['stim_category'] = df['stim_category'].apply(newcat2oldcat).astype(int)
df['resp_category'] = df['resp_category'].apply(newcat2oldcat).astype(int)

df = df.rename(columns={'subject_name': 'Subj_idx',
                   'stim_category': 'Stimulus',
                   'stim_orientation': 'Orientation',
                   'resp_category': 'Response',
                   'resp_rt': 'RT_decConf',
                   'resp_confidence': 'Confidence',
                   'stim_cue_validity_id': 'Difficulty'})
cols = ['Subj_idx', 'Stimulus', 'Response', 'Confidence', 'RT_decConf', 'Difficulty', 'Orientation']


df[cols].sort_values('Subj_idx').to_csv('for_rahnev_database/attention_doby.csv', index=False)

