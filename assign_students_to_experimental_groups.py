#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pandas import DataFrame as df
import pandas as pd
import numpy as np
import os
import sys
import datetime
import analysis
import matplotlib.pyplot as plt

import data_utils

#------------ init update variables ----------------:
OVERRIDE = True

initial_assignment=False
is_update_assignments=False
classes_to_update='all' #['SD (5.1)','SD (5.2)']

assignments_NOT_to_update=['Final assignment', 'Final assignment (predefined)']
if initial_assignment:
    assignments_NOT_to_update=[]

assignments_to_evaluate=['Final assignment','* Re-assign if more students are added']

# ---------- init var names and directories  ---------------:

new_pretest_scores_filename= '15.2.18.csv'
old_MD_users_filename= 'MD_users6_2.csv'
old_MD_classes_filename='MD_classes_2018-02-15.csv'
md_users_to_pretest_index_map = {'SchoolSign': 'sSchoolName', 'GroupID': 'class2', 'Role': 'sRole'} #
pretest_to_md_users_index_map= {v: k for k, v in md_users_to_pretest_index_map.items()}


#------------ init group assignments variables ----------------:

assignment_method='random_sampling_by_class_and_percentile'
groups_names = ['1','2','3'] #['A1', 'A2', 'B1', 'B2' #
divide_by_names={-1:'no_score',0:'s==0',1:'0<s<60',2:'s>60'}
groups_distribution = [.3, .3, .4] #[.2, .2, .3, .3] #
divide_by=[0.,1.,60.]# [0.25, 0.5, 0.75, 1] # if max value is =<1 assignment by percentiles, else by value.
special_education_classes=['AR (4.3)','AR (4.4)']
determenistic_assignments={'AR (4.3)':2.,'AR (4.4)':2., 'GB-1 (4.2)':'dropped', 'GB-1 (4.3)':'dropped'}
diff_threshold=3
n_groups=len(groups_distribution)

assignments_args=locals()

#------------ init directories and relevant columns ----------------:

ROOT=r'C:\Users\t-tatron\PycharmProjects\ILDCEDUDataScience\CET\SplitStudents'
data_dir= os.path.join(ROOT,'local_data') #TODO = save here all data files which contain students ID, DO NOT upload to git!
output_dir=os.path.join(ROOT, 'output')
local_output_dir=os.path.join(ROOT, 'local_output') #TODO = save here all output files which contain students ID, DO NOT upload to git!
data_index_col='UserID'
pretest_score_col='AdaptivePrequalTask Score' #'fCalculatedMark'#'fCalculatedMark'
pretest_index_col='gStudentID'
now=datetime.datetime.now()

#------------- load new scores and old MD files -------------#:

new_pretest_scores=pd.read_csv(os.path.join(data_dir, new_pretest_scores_filename),index_col=data_index_col)

new_pretest_scores.rename(columns = {pretest_score_col: 'pretest_score', new_pretest_scores.columns[0]: 'sSchoolName'}, inplace = True)
new_pretest_scores['pretest_score']=pd.to_numeric(new_pretest_scores['pretest_score'], errors='coerce')
new_pretest_scores.columns=new_pretest_scores.columns.astype(str)

old_MD_users = df.from_csv(os.path.join(data_dir, old_MD_users_filename), index_col=data_index_col) #the file to update
old_MD_classes=df.from_csv(os.path.join(data_dir, old_MD_classes_filename ), parse_dates=['start_date'])

old_MD_users = old_MD_users.rename(columns = md_users_to_pretest_index_map)

#-----------------classes----------------------------------
class GroupAssignment():

    def __init__(self,old_MD_users=old_MD_users, old_MD_classes=old_MD_classes, new_pretest_scores=None):
        """
        take old MD_users and update assignemts
        :param old_MD_users: last MD_users saved in Azure Storage and deployed for experiment
        :param old_MD_classes: last MD_classes file, with updated 'start_date', and '
        :param new_pretest_scores: new pretest_scores file sent from CET, processed and saved as CET.
        """
        self.assignment_args = assignments_args
        self.old_MD_classes = old_MD_classes
        self.old_MD_users = old_MD_users
        self.new_pretest_scores = new_pretest_scores
        self.new_MD_users = old_MD_users.loc[old_MD_users.IsTest==False].loc[old_MD_users.sRole=='student'].copy()
        self.new_MD_users['new_group_index'] = np.nan
        self.new_MD_users['class_percentile'] = np.nan
        self.new_MD_classes = old_MD_classes.copy()
        return

    def select_classes_to_update(self, classes_to_update, assignments_NOT_to_update):
        """
        select classes to update based on start date, filters and assignment status (final\not final)
        :param classes_to_update: if 'all' choose all calsses in the experiment.
        :param assignments_NOT_to_update:
        :return:
        """
        if classes_to_update == 'all':
            classes_to_update = self.old_MD_classes.index


        # make sure no class who started the experiment is updated:
        classes_who_started_the_experiment = self.old_MD_classes.loc[self.old_MD_classes['start_date'] < now].index
        #classes_with_final_assignment = self.old_MD_classes.loc[self.old_MD_classes.assignment_type in assignments_NOT_to_update].index

        classes_to_update_new = [c for c in classes_to_update if
                             c not in classes_who_started_the_experiment and
                             self.old_MD_classes.loc[c].assignment_type not in assignments_NOT_to_update and
                             c not in ['last_update', 'with_file','drive_file']]

        self.assignment_args['updated_classes']=classes_to_update_new
        self.updated_classes=classes_to_update_new
        return classes_to_update_new

    '''def preprocess_assignments(self, students_pretest_scores=None, students_metadata=None, load_assignments_from=None):
        if load_assignments_from is not None:
            processed_students_assignments=df.from_csv(os.path.join(local_output_dir,load_assignments_from))
            class_details = processed_students_assignments.groupby('class2').first()[
                ['assignment_type', '%scores_per_class']]
            if OVERRIDE:
                self.new_MD_classes['assignment_type']= class_details.assignment_type.loc[old_MD_classes.index]
                self.new_MD_classes['%scores_per_class']=class_details['%scores_per_class'].loc[old_MD_classes.index]
                self.new_MD_classes.at['last_update', 'assignment_type'] = now
                self.new_MD_classes.at['with_file', 'assignment_type'] = new_pretest_scores_filename

                self.new_MD_classes.to_csv(os.path.join(data_dir, 'MD_classes_%s.csv' %now.date()))

        else:
            processed_students_assignments=preprocess_students_assignment(students_pretest_scores, students_metadata)


        return processed_students_assignments'''

    def update_group_assignments(self, new_pretest_scores, update_type=assignments_NOT_to_update, update_classes=classes_to_update, OVERRIDE=False):
        """updates the selected assignemtns"""
        classes_to_update= self.select_classes_to_update(update_classes,update_type)
        percent_scores_per_class=evaluate_students_pretest_scores(new_pretest_scores, OVERRIDE=OVERRIDE)
        self.new_MD_classes['%_scores_per_class']= percent_scores_per_class

        self.new_MD_classes.at['last_update', 'assignment_type'] = now
        self.new_MD_classes.at['with_file', 'assignment_type'] = new_pretest_scores_filename
        self.new_MD_classes.at['drive_file', 'assignment_type'] = self.old_MD_classes.at['drive_file', 'assignment_type']

        self.new_MD_users['pretest_score']=new_pretest_scores.pretest_score.loc[self.new_MD_users.index]

        print('\n\n ----------updating students assignments...-----------')
        for class2, class_students in self.new_MD_users.groupby('class2'):
            print(class2)
            print(class_students.head())
            self.new_MD_users.loc[class_students.index]['start_date'] = self.old_MD_classes.loc[class2]['start_date']
            if class2 in classes_to_update:
                if class2 in determenistic_assignments.keys():
                    predefined_assignment=self.assignment_args['determenistic_assignments'][class2]
                    self.new_MD_users['new_group_index'].loc[class_students.index]= predefined_assignment
                else:
                    #class_students=pd.concat([class_students, new_pretest_scores.pretest_score.loc[class_students.index]], axis=1)
                    print('--updated--')
                    print(class_students.head())
                    output = self.assign_groups(class_students, assignment_method)
                    self.new_MD_users['new_group_index'].loc[class_students.index] = output['group']
                    self.new_MD_users['class_percentile'].loc[class_students.index] = output['class_percentile']

        full_output = self.adjust_assignment_to_output_file(self.new_MD_users)

        if OVERRIDE:
            self.new_MD_classes.to_csv(os.path.join(data_dir, 'MD_classes_%s.csv' % now.date()))
            full_output.to_csv(os.path.join(data_dir, 'MD_users_%s.csv' % now.strftime("%Y%m%d-%H%M%S")))

        return self.new_MD_users, self.new_MD_classes

    def adjust_assignment_to_output_file(self , updated_assignments):
        ''' fit the output file (MD_users) to the format needed for deployment'''
        output = updated_assignments[['sSchoolName', 'class2', 'sRole', 'new_group_index','ExpGroup','start_date','IsTest']].copy()
        updated_students_group=output['new_group_index'].dropna()
        output['ExpGroup'].loc[updated_students_group.index] = updated_students_group.apply(
            lambda i: groups_names[int(i)])
        output.drop('new_group_index',axis=1, inplace=True)

        #output['start_date'] = ''
        output.index.name = data_index_col
        print(output.head())

        old_output_index = set(self.old_MD_users.index).difference(set(output.index))
        special_education_students = []
        for class2, c_students in output.groupby('class2'):
            #output['start_date'].loc[c_students.index] = self.old_MD_classes.loc[class2]['start_date']
            if class2 in special_education_classes:
                output['sRole'].loc[c_students.index] = 'special_education_student'
                # output.at['sRole', c_students]= 'special_education_student'
        # output['ExpGroup'].loc[special_education_students]=groups_names[2]
        #output['IsTest'] = False
        # add new output to old output:
        full_output = pd.concat([output, self.old_MD_users.loc[old_output_index]])

        full_output = full_output.rename(columns=pretest_to_md_users_index_map)
        full_output.index.name = data_index_col
        # full_output.columns=old_output.columns
        print('----------classes which were updated: %s-------------' % str(self.updated_classes))
        self.output = full_output

        return full_output

    def assign_groups(self, students, method='random_sampling_by_class_and_percentile'):
        ''' assigns students to experimental groups based on their class, class and percentile (deafult), or randomly'''
        if method=='naive_sampling':
            print('-------------- Grouping results  - Naive sampling ------------------')
            groups=random_assignment(students.index, group_dist=groups_distribution)

            #observed,expected=self.evaluate_group_assignment(students, test_vars='class2', groups_names =groups_names)

            return groups

        elif method == 'random_sampling_by_class':
                print('-------------- Grouping results  - Naive sampling by class ------------')
                students_group=pd.Series(index=students.index)
                student_score_percentile=pd.Series(index=students.index, name='class_percentile')
                students['overall_percentile']=0
                last_diff = np.random.randint(n_groups)
                for c, c_students in students.groupby('class2'):
                    students=c_students.index
                    print('class name:  %s ' %c_students[0])
                    students_group[students],last_diff = random_assignment(students, group_dist=groups_distribution, assign_diff_by=last_diff)
                    student_score_percentile[students] = calc_student_percentile(students.pretest_score.loc[students],
                                                                           by='pretest_score',percentiles=divide_by)
                    print('---------')
                #overall_percentiles = calc_student_percentile(students.pretest_score,by='pretest_score', )
                output=pd.concat([students_group, student_score_percentile],axis=1)
                output.columns=['group','class_percentile']

                return output

        elif method == 'random_sampling_by_class_and_percentile':

                students_group=pd.Series(index=students.index)
                student_score_percentile=pd.Series(index=students.index, name='class_percentile')
                for class_students in students.groupby('class2'):
                    c_students=class_students[1].index
                    print('class name:  %s ' %class_students[0])
                    student_score_percentile[c_students] = calc_student_percentile(students.loc[c_students],by='pretest_score')
                    last_diff = np.random.randint(n_groups) #
                    for p in student_score_percentile[c_students].groupby(by=student_score_percentile):
                        print('**** percentile = %i ****' %p[0])
                        p_students=p[1].index
                        #print('**last_diff=%f' % last_diff)
                        students_group[p_students], last_diff = random_assignment(p_students, group_dist=groups_distribution, assign_diff_by=last_diff)
                        print('next group assignment = %i' %last_diff)
                    print('---------')
                    #fix imbalanced samples:
                    #students_group=fix_unbalanced_samples(students, students_group, student_score_percentile, group_distribution)

                output = pd.concat([students_group, student_score_percentile],axis=1)
                output.columns = ['group', 'class_percentile']

                return output

    def group_and_filter_students(self, MD_users, pretest_scores=None, return_cols='all', filter_by='class2', filters=['all'], add_pretest_scores=False):
        ''' apply filters on students metadata (MD_users), deafult filter by class == all '''
        MD_users= MD_users.loc[MD_users.IsTest == False]
        MD_users = MD_users.loc[MD_users.Role == 'student']
        if add_pretest_scores:
            MD_users = pd.concat([MD_users, pretest_scores['pretest_score']], axis=1).loc[MD_users.index]
            student_score_percentile = pd.Series(index=MD_users.index, name='class_percentile')
            for c, c_students in MD_users.groupby(filter_by):
                print('class name:  %s ' % c)
                student_score_percentile[c_students.index] = calc_student_percentile(c_students,
                                                                               by='pretest_score')
            MD_users['class_percentile'] = student_score_percentile
        print('--------- data filtered by %s = %s --------------' %(filter_by,str(filters)))
        if return_cols=='all':
            return_cols=MD_users.columns
        filtered_data=df()
        for i,students in MD_users.groupby(filter_by):
            if filters==['all']:
                filtered_data = pd.concat([filtered_data, students])
            elif i in filters:
                filtered_data=pd.concat([filtered_data, students])

        return filtered_data[return_cols]

    def evaluate_group_assignments(self, grouped_data, grouping_var='ExpGroup', max_diff=diff_threshold, test_vars=['class2'],
                                   ):
        '''evaluates students asssignments by performing Chi^2 test, and by checking the difference
        between expected and observed number of students in each experimental group'''
        print('\n')

        if type(test_vars) == str:
            test_vars = [test_vars]
        for v in test_vars:
            print("---------%s EVALUATION-------------" % v.upper())
            observed_counts, expected_counts, chi2, p, dof, conclusion = data_utils.chi2_test(grouped_data, grouping_var, v)
            if groups_names is not None:
                observed_counts.index = groups_names
                expected_counts.index = groups_names
            print("---------observed %s-%s distribution -------------" % (grouping_var, v))
            print(observed_counts.T.applymap(int))
            print("---------expected %s-%s distribution -------------" % (grouping_var, v))
            print(df(expected_counts, index=observed_counts.index, columns=observed_counts.columns).T)
            print("--------------------------------")
            print(conclusion)
            plt.figure()
            p_observed_counts=observed_counts.copy()
            p_observed_counts.index=groups_names
            if v=='class_percentile':
                p_observed_counts=p_observed_counts.rename(columns= divide_by_names)
            ax = p_observed_counts.plot(kind='bar', colormap='tab20b', figsize=(8, 5),
                                      title='Student count in each group by %s' % v)
            ax.legend(loc='best')
            del p_observed_counts
            observed_diff = observed_counts - expected_counts

            differences = observed_diff.unstack().apply(round, 3)

            conclusions = ''
            for i, p in enumerate(ax.patches):
                diff = differences.iloc[int(i)]
                if diff > max_diff or diff < -max_diff:
                    if diff > max_diff:
                        conclusion = '%s is OVER represented in group %s (+%s)\n' % (
                            differences.index[i][0], differences.index[i][1], diff)

                        conclusions = conclusions + conclusion
                    elif diff < -max_diff:
                        conclusion = '%s is UNDER represented in group %s (%s)\n' % (
                            differences.index[i][0], differences.index[i][1], diff)
                        conclusions = conclusions + conclusion
                    # ax.text(patch.get_x() - .03, patch.get_height() + .5, \str(diff), fontsize=10,color='r')
                    ax.annotate(str(diff), (p.get_x() * 1.005, p.get_height() * 1.005), color='r')
                else:
                    ax.annotate(str(diff), (p.get_x() * 1.005, p.get_height() * 1.005), color='k')

            if OVERRIDE:
                plt.savefig(os.path.join(output_dir, '%s results_evaluation_%s.png' % (now.strftime("%Y%m%d-%H%M%S"),v)))
            plt.close()
            plot_grouped_data(grouped_data)
            if conclusions == '':
                conclusions = 'All %s values are represented in a balanced manner (max difference<%i)' % (v, max_diff)
            print("--------------------------------")
            print(conclusions)
            print("--------------------------------")

        return observed_diff, observed_counts

#-----------------functions--------------------------------
def calc_student_percentile(students_scores, by='pretest_score', percentiles=divide_by):
    """ assign each score with its corresponding percentile
    - if max(percentiles) <=1 --> divide by percentiles (e.g. 0,0.5,1)
    - elif max(percentiles)> 1 --> divide by absolute values (e.g. 0, 60,100) """

    if max(percentiles)>1:
        q_scores=divide_by
    else: #calc score_threshold_by predefined percentiles percentiles
        q_scores=students_scores[by].quantile(percentiles,interpolation='lower')

    students_percentile=pd.Series(0,index=students_scores.index, name='percentile')
    for q in q_scores:
        print(q)
        students_percentile.loc[students_scores[by]>q]+=1
    students_percentile.loc[students_scores[by].isnull()]=-1
    #students_percentile=[percentiles[p-1] for p in students_percentile]
    ss=pd.concat([students_percentile,students_scores],axis=1)
    ss_desc = ss.groupby('percentile').describe()['pretest_score'].unstack()
    ss_desc=ss_desc[['count','mean','std','min','max']]
    print(q_scores)
    #print(students_scores)
    print(students_percentile)
    print(ss_desc)
    #percentiles_ind=[i for i in ss_desc.index]
    #ss_desc.index = list(percentiles[percentiles_ind])+['no_score']
    #print(ss_desc)

    return students_percentile

def random_assignment(samples_list, n_groups=None, group_dist=groups_distribution, assign_diff_by='by_dist_no_replacement'):
    ''' randomly assign samples to n_groups according to a given group distribution, used in the 'assign group' funciton '''
    print('number of students: %i' %len(samples_list))
    #print('group_distribution: %s ' %group_dist)

    n=len(samples_list)

    if n_groups is None and group_dist is not None:
        n_groups=len(group_dist)
    elif n_groups!=len(group_dist):
        print('ERROR  - number of groups does not match group distribution!')
        return
    elif sum(group_dist)!=1:
        print('ERROR  - the group distribution does not sum to 1!')

    groups_sizes=[int(np.floor(d*n)) for d in group_dist]

    diff=n-sum(groups_sizes)
    print('group distribution: %s (%i missing)' %(groups_sizes,diff))
    last_diff=assign_diff_by
    if diff>0:
        #if the division to groups has mod>0 - assign the 'redundant' students randomly or systematically to groups.
        #print('diff == %i ' %diff)

        if assign_diff_by== 'random_no_replacement': #how to assign 'redundant' students to groups
            diff_groups=np.random.choice(range(n_groups), size=diff, replace=False)
        elif assign_diff_by == 'by_dist_no_replacement':
            diff_groups = np.random.choice(range(n_groups), size=diff, replace=False, p=group_dist)
        elif type(assign_diff_by)==int: #assign students based on previous assignments (each time start with a different group)
            diff_groups=range(assign_diff_by, assign_diff_by + diff+1)
            diff_groups=[d if d in range(n_groups) else d-n_groups for d in diff_groups]
            last_diff = diff_groups[-1]
            diff_groups=diff_groups[:-1]

            #print('***students were asigned to groups : %s' %diff_groups)



        for i in diff_groups:
            groups_sizes[i] += 1 #adds students to the randomly\systematically selected groups

        print('adjusted group distribution: %s ' % groups_sizes)
        #print('n: %s' %sum(groups_sizes))
    grouping=[[i]*groups_sizes[i] for i in range(n_groups)]
    grouping=np.array([val for sublist in grouping for val in sublist])
    np.random.shuffle(grouping)

    group_assignments=pd.Series(grouping,name='group')
    group_assignments.index=samples_list
    if type(assign_diff_by)==int:
        return group_assignments,last_diff
    return group_assignments

def balance_group_assignment(grouped_data, fix_threshold=diff_threshold, grouping_var='group', grouping_var_names=groups_names, block_var='class2', fix_var='overall_percentile', fix_var_names=divide_by+['missing'], test_vars=['class2', 'class_percentile']):
    '''in case the groups were not balanced, change assignments to achieve balanced sampling'''
    print('\n\n----- BALANCING GROUP ASSIGNMENTS (fix_threshold = %i) ----------' %fix_threshold)
    new_groups=grouped_data[grouping_var].copy()
    max_diff=fix_threshold
    iterations=0
    while max_diff>=fix_threshold or iterations<10:
        print('------iteration: %i------' %iterations)
        iterations+=1
        observed_counts, expected_counts, chi2, p, dof, conclusion = data_utils.chi2_test(grouped_data, 'group', fix_var)
        observed_diff = df(observed_counts - expected_counts, index=set(grouped_data[grouping_var]),
                           columns=set(grouped_data[fix_var]))
        #print data frames:
        data_utils.df_print(observed_diff, index=grouping_var_names, columns=fix_var_names, title='**observed diff:**')
        data_utils.df_print(observed_counts, index=grouping_var_names, columns=fix_var_names, title='**observed counts:**')


        observed_diff=observed_diff.unstack().sort_values(ascending=False)
        #find non - balanced groups, take the maximum non balanced group:
        max_diff=max(observed_diff.apply(abs))

        if any(observed_diff==max_diff): #its a positive value:
            d = observed_diff[observed_diff==max_diff]
            group1=d.index[0][1]
            #move from the group with maximum students
            fix = d.index[0][0]  # the value which should be removed from group1

            # find a group with minimum number of 'fix1'
            fix_diff = observed_diff[fix]
            group2 = fix_diff[fix_diff == fix_diff.min()].index[0] # find a group missing fix1

        else: #its a negative value
            d = observed_diff[observed_diff == - max_diff]
            group2 = d.index[0][1]
            fix = d.index[0][0]  # the value which should be removed from group1
            fix_diff = observed_diff[fix]
            group1 = fix_diff[fix_diff == fix_diff.max()].index[0]

        group1_students = grouped_data.loc[grouped_data[grouping_var] == group1]
        group2_students = grouped_data.loc[grouped_data[grouping_var] == group2]
        #find a specific student:
        exchange_students=group1_students.loc[group1_students[fix_var] == fix]
        exchange=False
        i=0
        while exchange==False and i<10:
            student1=exchange_students.sample(n=1)
            block=student1[block_var].values[0]
            #if the number of students in class is bigger in group1:
            if group1_students[block_var].value_counts()[block] <group2_students[block_var].value_counts()[block]:
                #move the student from group1 to group2
                new_groups.loc[student1.index]=group2
                print('--Exchanged students:--')
                print('* %s  : %i-->%i' % (grouping_var, group1, group2))
                print('* %s = %s' % (block_var, block))
                print('* %s = %s' % (fix_var, fix))
                exchange=True
            i+=1 #try for 10 iterations
            grouped_data[grouping_var] = new_groups

    print('\n\n')
    return grouped_data

def evaluate_students_pretest_scores(students_pretest_scores, OVERRIDE=False):
    ''' describe the number of students who have/missing pretest scores in the file sent from CET.'''
    make_int_str = lambda DF: DF.fillna(-1).applymap(lambda s: '' if s == -1 else str(int(s)))
    make_percent_str = lambda DF: DF.fillna(-1).applymap(lambda s: '' if s == -1 else '%i' % (s * 100) + '%')
    # describe students scores
    if OVERRIDE:
        # plot pretest score histogram
        students_pretest_scores['pretest_score'].plot(kind='hist', bins=100)
        plt.title('students scores distribution')
        plt.xlabel('Pre-test score')
        plt.ylabel('Number of students')
        plt.savefig(os.path.join(output_dir, 'pretest_scores_hist_%s.png' %now.date()))
        plt.close()

    scores_desc = count_data(students_pretest_scores, value='pretest_score', index='sSchoolName', column='Class')
    data_desc = count_data(students_pretest_scores, index='sSchoolName', column='Class')
    missing_scores = [i for i in data_desc.index if i not in scores_desc.index]
    scores_desc = pd.concat([scores_desc, data_desc.loc[missing_scores].applymap(lambda s: 0 if s > 0 else np.nan)])
    scores_desc[scores_desc.fillna(-1) < 0] = 0
    scores_percent = scores_desc / data_desc
    scores_relative_desc = make_int_str(scores_desc) + '/' + make_int_str(data_desc) + ' (' + make_percent_str(
        scores_percent) + ')'
    scores_relative_desc[data_desc.fillna(0) == 0] = ''

    print('\n---- number of pre-test scores from class students ---- :')
    print(scores_relative_desc)
    scores_percent = scores_percent.unstack()
    full_classes = scores_percent.loc[scores_percent > 0.5]
    full_classes.index = full_classes.index.swaplevel(1, 0)

    # missing_classes=students_metadata.loc[students_metadata['pretest_score'].fillna(-1)<=0]['class2']
    missing_classes = []
    missing_students = students_pretest_scores.loc[students_pretest_scores['pretest_score'].fillna(-1) <= 0].sort_values(
        by='class2')
    for class_name, class_students in missing_students.groupby(['sSchoolName', 'Class']):
        if class_name not in list(full_classes.index):
            missing_classes.append(class_name)

    print('\n --- %i missing_classes (less than 0.5 of the students) :  ---' % len(missing_classes))

    #print(missing_students)

    print('\n --- scores description --- :')
    print(students_pretest_scores.describe())
    ('\n --- distdibution over schools and classes : ')

    students_scores_mean = df.pivot_table(students_pretest_scores, values='pretest_score', index='class2')
    students_scores_std = df.pivot_table(students_pretest_scores, values='pretest_score', index='class2', aggfunc='std')
    print(students_scores_mean)
    print(students_scores_std)
    percent_score_per_class = scores_relative_desc.unstack().replace('', np.nan).dropna()
    percent_score_per_class.index= ['%s (%s)' % (i[1], i[0]) for i in percent_score_per_class.index]
    if OVERRIDE:
        scores_relative_desc.to_csv(os.path.join(local_output_dir, 'scores_percent_per_class_%s.csv' %now.date()))

    return percent_score_per_class

def preprocess_students_assignment(students_pretest_scores, students_metadata, OVERRIDE=False):
    '''
    Evaluate the scores and the metadata files to test which students need to be assigned to which group.
    #There are 5 different cases:
    #(1) classes who have determenistic assignment (special education schools - AR 4.3, AR 4.4)
    #(2) classes for which there are no pretest scores (<50%) --> assign to ADL** (two stars = have to be changed!)
    #(3) schools\students with no class information --> assign to ADL** or randomly assign
    #(4) students who have prescores and also their classes (>50%)--> assigned semi-randomly to groups* (one star = possible to change if there is more information)
    #(5) students who don't have pretest scores, but their class have (>50%) --> assigned randomly to groups*
    #(6) students in class with 100% pretest scores --> assigned semi-randomly to groups (no star(:)

    :return:
    new_concated_students_table - including:
    'experimental group assignment' -
        a group value, - A1, A2, B1, B2 marked with * or ** if temporary
        'random_sampling_by_class_and_percentile' (to use  - method= 'random_sampling_by_class_and_percentile')
        'random-sampling' (to use - method = 'naive sampling' )

    '''


    print('------------PREPROCESSING STUDENTS ASSIGNMENTS ----------')
    students_scores = data_utils.remove_duplicates(students_pretest_scores, keep='first')
    students_metadata['pretest_score'] = students_scores.pretest_score.loc[students_metadata.index]
    students_metadata.Class = students_metadata.Class.apply(str)
    joint_columns = set(students_scores.columns).intersection(students_metadata.columns)
    students_scores.columns = [s if s not in joint_columns else s + '0' for s in students_scores.columns]
    concated_students_table = pd.concat([students_scores, students_metadata], axis=1)

    description = df(index=['group', 'assignment_method', 'assignment_type', '%scores_per_class','experiment_start_date'],
                     columns=concated_students_table.index)

    # if there is no information about students classes:
    missing_class_students = concated_students_table.loc[concated_students_table.class2.isnull()]
    for s, s_students in missing_class_students.groupby('sSchoolName'):
        percent_pretest_scores = len(s_students.pretest_score.dropna()) / len(s_students)
        if s in determenistic_assignments.keys():
            class_assignment_type = 'Final assignment (predefined)'
            class_assignment = determenistic_assignments[s]
            class_method = 'Predefined'
        else:
            class_assignment_type = '** Temporary assignment (missing class information)'
            class_assignment = ''
            class_method='random_sampling_by_class_and_percentile'
        desc = pd.Series([class_assignment, class_method, class_assignment_type,
                          '%.2f' % percent_pretest_scores], index=description.index)
        for s in s_students.index:
            description[s] = desc

    for c, c_students in concated_students_table.groupby('class2'):
        class_start_date = old_MD_classes.loc[c]
        if c in determenistic_assignments.keys():
            class_assignment_type = 'Final assignment (predefined)'
            class_assignment = determenistic_assignments[c]
            class_method = 'Predefined'
        else:
            percent_pretest_scores = len(c_students.pretest_score.dropna()) / len(c_students)
            if percent_pretest_scores > .5:
                class_assignment_type = '* Re-assign if more students are added'
                class_assignment = ''
                class_method = 'random_sampling_by_class_and_percentile'
            if percent_pretest_scores < .5:
                class_assignment_type = '** Temporary assignment (missing class scores)'
                class_assignment = ''
                class_method = 'random_sampling_by_class_and_percentile'
            if percent_pretest_scores == 1:
                class_assignment_type = 'Final assignment'
                class_assignment = ''
                class_method= 'random_sampling_by_class_and_percentile'

        desc = pd.Series([class_assignment, class_method, class_assignment_type,
                          '%.2f' % percent_pretest_scores, class_start_date], index=description.index)
        for c in c_students.index:
            description[c] = desc

    print('\n----- pre-assignment summary : -------- ')
    print(description.T['assignment_type'].value_counts())

    new_concated_students_table = pd.concat([concated_students_table, description.T], axis=1)
    print('\n-------percent of pretest scores : -----------')
    assignment_report = new_concated_students_table.groupby('class2').first()[['assignment_type', '%scores_per_class']]
    new_MD_classes['assignment_type']=assignment_report.assignment_type.loc[old_MD_classes.index]
    new_MD_classes['%scores_per_class']=assignment_report['%scores_per_class'].loc[old_MD_classes.index]
    new_MD_classes.at['last_update', 'assignment_type']=now
    new_MD_classes.at['with_file', 'assignment_type'] = new_pretest_scores_filename
    assignment_report2 = new_concated_students_table.loc[missing_class_students.index].groupby('sSchoolName').first()[
        ['assignment_type', '%scores_per_class']]

    new_concated_students_table['last_update'] = str(datetime.datetime.today())
    new_concated_students_table['class_percentile']=''
    new_concated_students_table['overall_percentile']=''

    print(assignment_report)
    print(assignment_report2)

    if OVERRIDE:
        new_concated_students_table.to_csv(os.path.join(local_output_dir, 'students_assignments_desc_%s.csv' %now.date()))
        new_MD_classes.to_csv(os.path.join(data_dir, 'MD_classes_%s.csv' %now.date()))
    return new_concated_students_table

def add_LO_indexes_to_meta_data(meta_data=None, lo_index=None, is_load_data=True, md_file='MD_math_processed.csv', loi_file='LOs_order_index_fraction.csv', filter_by_language=True, save_to_csv=True, save_name='MD_math_processed.csv'):
    """
    adds the Learning Objective index to questions metadata, in order to know the order of questions.
    Aslo adds 'question index' - the order in which the questions appeared in CET content player using LO index + the number of question in LO session.

    :param meta_data:
    :param lo_index:
    :param is_load_data:
    :param md_file:
    :param loi_file:
    :param filter_by_language:
    :param save_to_csv:
    :param save_name:
    :return:
    """
    if is_load_data:
        meta_data= df.from_csv(os.path.join(DATA_ROOT,md_file))
        lo_index = df.from_csv(os.path.join(DATA_ROOT,loi_file))
    # lo_index.index = [s.lower() for s in lo_index.index]
    meta_data = meta_data.loc[meta_data.nLanguage == 1]
    meta_data = meta_data.reset_index(drop=True)
    lo_index_ordered = lo_index.loc[meta_data.gLO].reset_index(drop=True)
    meta_data[lo_index_ordered.columns] = lo_index_ordered
    if 'num_of_questions_in_lo_session' in meta_data.columns:
        meta_data.num_of_questions_in_lo_session=meta_data['sQuestionPageID'].apply(lambda s: int(s[s.rfind('_') + 1:]))

    meta_data['question_index'] = meta_data.LO_general_index + 0.01 * meta_data.num_of_questions_in_lo_session

    if filter_by_language:
        meta_data = meta_data.loc[meta_data.nLanguage == 1]

    meta_data.drop_duplicates(inplace=True)
    if save_to_csv:
        meta_data.to_csv(os.path.join(DATA_ROOT,save_name))
    return meta_data


if __name__ == "__main__":
    log_file = os.path.join(output_dir, "%s_report.txt" % now.date())
    with open(log_file, "w") as f: #write a txt report
        if OVERRIDE:
            sys.stdout = f
        print(datetime.datetime.now())
        print('\n')

        if initial_assignment: #first pretest_scores file - if it doesn't work, 'update_assignments' based on pseudo MD_users file.
            processed_assignments = preprocess_students_assignment(students_pretest_scores, students_new_pretest_scores, OVERRIDE=OVERRIDE)
            processed_assignments.to_csv(os.path.join(local_output_dir, 'student_assignment_initial_%s.csv' %now.date()))
        #evaluate_students_pretest_scores(students_pretest_scores, students_metadata,OVERRIDE=OVERRIDE)

        GA = GroupAssignment(old_MD_users=old_MD_users, old_MD_classes=old_MD_classes)
        if is_update_assignments:
            #percent_students_per_class=evaluate_students_pretest_scores(students_pretest_scores, students_new_pretest_scores, OVERRIDE=OVERRIDE)
            #old_assignments_desc = GA.preprocess_assignments(load_assignments_from=old_assignments_desc_filename)
            GA.update_group_assignments(new_pretest_scores, OVERRIDE=OVERRIDE )
            updated_assignments=GA.new_MD_users
            grouped_data = GA.group_and_filter_students(updated_assignments, return_cols='all', filter_by='GroupID',
                                                        filters=GA.updated_classes)
        else:
            updated_assignments=df.from_csv(os.path.join(data_dir,old_MD_users_filename))
            grouped_data=GA.group_and_filter_students(updated_assignments, return_cols='all', filter_by='GroupID', add_pretest_scores=True, pretest_scores=new_pretest_scores)# NZ (4.1)', 'NZ (4.2)', 'SP (4.2)', 'SP (4.3)'])

        GA.evaluate_group_assignments(grouped_data, test_vars=['GroupID', 'class_percentile']) #rig



    '''students_metadata_grouped=df.from_csv(os.path.join(output_dir,'random_sampling_by_class_groups.csv'))
        students_metadata_grouped=balance_group_assignment(students_metadata_grouped)
        describe_grouped_data(students_metadata_grouped)
        evaluate_group_assignment(students_metadata_grouped, test_vars=['class2', 'class_percentile', 'overall_percentile'],OVERRIDE=True)
        calc_student_percentile(students_scores)'''

#-----------------execute--------------------------------







