import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
plt.style.use('fivethirtyeight')

def connect_to_sql_database(database_name='database.sqlite3'):
    conn = sqlite3.connect(database_name)
    c = conn.cursor()
    return conn, c

def sql_to_pandas_df(query, cursor):
    cursor.execute(query)
    df = pd.DataFrame(cursor.fetchall())
    df.columns = [x[0] for x in cursor.description]
    df.head()

    return df

def convert_grades_to_ints(df):
    grades_list = ["a", "ab", "b", "bc", "c", "d", "f"]
    for grade in grades_list:
        df["%s_count" % grade] = df["%s_count" % grade].astype(int)

def add_total_grades(df):
    df["total_grades"] = df["a_count"] + df["ab_count"] + df["b_count"] + \
    df["bc_count"] + df["c_count"] + df["d_count"] + df["f_count"]

def filter_no_grades(df):
    return df[df["total_grades"] > 0]

def filter_only_one_instructor(df):
    grouped = df.groupby("course_uuid")
    return grouped.filter(lambda x: len(x["instructor_id"].unique())>1)

def compute_pmf_and_cdf(df, columns):
    pmf = np.array(df[columns].sum().values, dtype=float)
    pmf /= float(np.sum(pmf))
    cdf = np.cumsum(pmf)
    
    return pmf, cdf

def find_max_difference_in_cdfs(all_cdfs):
    all_cdfs = np.array(all_cdfs)
    min_cdf = np.min(all_cdfs, axis=0)
    max_cdf = np.max(all_cdfs, axis=0)
    cdfs_diffs = max_cdf-min_cdf
    max_diff = np.max(cdfs_diffs)

    return max_diff

def permute_columns_in_dataframe(df, columns):
    df[columns] = np.random.permutation(df[columns])

def compare_instructor_grade_distributions_by_permutation(df, npermutations=100, letter_grades = ['f_count', 'd_count', 'c_count', 'bc_count', 'b_count', 'ab_count', 'a_count'], load_existing_pickle=True, print_progress=True):
    if load_existing_pickle:
        if os.path.isfile(f"all_pvalues_{npermutations}.pkl"):
            all_pvalues = pickle.load(open(f"all_pvalues_{npermutations}.pkl", "rb"))
        else:
            all_pvalues = {}
    else:
        all_pvalues = {}

    grouped_courses = df.groupby("course_uuid")
    print(f"{len(grouped_courses)} unique courses found")
    for course_index, (course, course_df) in enumerate(grouped_courses): 
        if course in all_pvalues.keys():
            continue
        grouped_course_offerings = course_df.groupby("instructor_id")
        num_instructors = len(grouped_course_offerings)
        all_cdfs = []
        for instructor, instructor_df in grouped_course_offerings:
            pmf, cdf = compute_pmf_and_cdf(instructor_df, letter_grades)
            all_cdfs.append(cdf)
        real_max_diff = find_max_difference_in_cdfs(all_cdfs)

        permuted_max_diffs = []
        for _ in range(npermutations):
            permute_columns_in_dataframe(course_df, ["instructor_id", "instructor_name"])
            grouped_course_offerings = course_df.groupby("instructor_id")
            num_instructors = len(grouped_course_offerings)
            all_cdfs = []
            for instructor, instructor_df in list(grouped_course_offerings):
                pmf, cdf = compute_pmf_and_cdf(instructor_df, letter_grades)
                all_cdfs.append(cdf)
            max_diff = find_max_difference_in_cdfs(all_cdfs)
            permuted_max_diffs.append(max_diff)
        permuted_max_diffs = np.array(permuted_max_diffs)
        p_value = np.mean(permuted_max_diffs >= real_max_diff)
        
        if print_progress:
            print(course_index/float(len(grouped_courses)), course, p_value, num_instructors)
        all_pvalues[course] = (p_value, num_instructors)
        pickle.dump(all_pvalues, open(f"all_pvalues_{npermutations}.pkl", "wb"))
    
    return all_pvalues

def plot_instructor_cdfs(df, course_uuid, letter_grades = ['f_count', 'd_count', 'c_count', 'bc_count', 'b_count', 'ab_count', 'a_count'], npermutations=3, alpha=0.01, plot_all_permuted_cdfs=False, plot_permutation_mean_and_std=False, plot_permutation_distribution=False, plot_original_cdfs=True):
    if plot_original_cdfs:
        plt.figure()
    course_df = df[df["course_uuid"] == course_uuid]
    grouped_course_offerings = course_df.groupby("instructor_id")
    num_instructors = len(grouped_course_offerings)
    all_cdfs = []
    for instructor, instructor_df in grouped_course_offerings:
        pmf, cdf = compute_pmf_and_cdf(instructor_df, letter_grades)
        if plot_original_cdfs:
            plt.plot(cdf, label=instructor_df["instructor_name"].unique()[0])
        all_cdfs.append(cdf)

    if plot_original_cdfs:
        plt.xticks(range(8), ["F", "D", "C", "BC", "B", "AB", "A"])
        plt.legend(loc=(1.0, 0.5))

    real_max_diff = find_max_difference_in_cdfs(all_cdfs)
    permuted_max_diffs = []
    for _ in range(npermutations):
        permute_columns_in_dataframe(course_df, ["instructor_id", "instructor_name"])
        grouped_course_offerings = course_df.groupby("instructor_id")
        num_instructors = len(grouped_course_offerings)
        all_cdfs = []
        for instructor, instructor_df in list(grouped_course_offerings):
            pmf, cdf = compute_pmf_and_cdf(instructor_df, letter_grades)
            if plot_all_permuted_cdfs:
                plt.plot(cdf, color='gray', alpha=alpha)
            all_cdfs.append(cdf)
        all_cdfs = np.array(all_cdfs)
        max_diff = find_max_difference_in_cdfs(all_cdfs)
        permuted_max_diffs.append(max_diff)
    mean_cdf = np.mean(all_cdfs, axis=0)
    std_cdf = np.std(all_cdfs, axis=0)
    if plot_permutation_mean_and_std:
        plt.errorbar(["F", "D", "C", "BC", "B", "AB", "A"], mean_cdf, yerr=4*std_cdf, linewidth=2.0, color='black')
    permuted_max_diffs = np.array(permuted_max_diffs)
    p_value = np.mean(permuted_max_diffs >= real_max_diff)
    print(p_value)
    if plot_permutation_distribution:
        plt.figure()
        sns.distplot( permuted_max_diffs, norm_hist=False, kde=False, hist=True, color="red", label="Differences")
        plt.vlines(real_max_diff, ymin=0, ymax=10)
        plt.xlabel("Difference")
        plt.ylabel("Counts")
        plt.show()

def filter_duplicate_course_offering_uuids(df):
    df = df.drop_duplicates(subset=["course_offering_uuid"])
    return df

def make_combined_instructor_ids_for_team_teachers(df):
    groups_list = list(df.groupby("course_offering_uuid"))
    combined_instructor_id_dict = {}
    combined_instructor_name_dict = {}
    for i in range(len(groups_list)):
        combined_instructor_id = '-'.join(sorted(groups_list[i][1]["instructor_id"].unique()))
        combined_instructor_id_dict[groups_list[i][0]] = combined_instructor_id
        combined_instructor_name = '-'.join(sorted(groups_list[i][1]["instructor_name"].unique()))
        combined_instructor_name_dict[groups_list[i][0]] = combined_instructor_name
    df["instructor_id"] = [combined_instructor_id_dict[x] for x in df["course_offering_uuid"]]
    df["instructor_name"] = [combined_instructor_name_dict[x] for x in df["course_offering_uuid"]]

def get_only_lecture_section(df):
    df = df[df["section_type"] == "LEC"]
    return df

def plot_grade_distribution(grades, barplot=True, lineplot=False, distribution_type='pdf'):
    plt.figure(figsize=(8,6))
    grades = np.array(grades, dtype=float)
    grades /= np.sum(grades)
    if distribution_type == 'cdf':
        grades = np.cumsum(grades)
    if barplot:
        plt.bar(["F", "D", "C", "BC", "B", "AB", "A"], grades)
    if lineplot:
        plt.plot(grades, color='black')
        plt.xticks(range(8), ["F", "D", "C", "BC", "B", "AB", "A"])
    if distribution_type == 'pdf':
        plt.title("Grade probability density function", fontsize=30)
        plt.ylabel("PDF", fontsize=30)
    elif distribution_type == 'cdf':
        plt.title("Grade cumulative density function", fontsize=30)
        plt.ylabel("CDF", fontsize=30)
    plt.xlabel("Grade", fontsize=30)

def get_pvalues_by_subject(df, results):
    pvalue_by_subject_dict = {}
    for result in results.items():
        course_uuid, (pvalue, num_instructors) = result
        subject_name = df[df["course_uuid"] == course_uuid]["subject_name"].iloc[0]
        if subject_name in pvalue_by_subject_dict.keys():
            pvalue_by_subject_dict[subject_name].append(pvalue)
        else:
            pvalue_by_subject_dict[subject_name] = [pvalue]
    
    return pvalue_by_subject_dict

def bootstrap_confidence_intervals_for_mean(data, n_resample=1000, alpha=0.05):
    means = []
    for _ in range(n_resample):
        means.append(np.mean(np.random.choice(data, len(data))))
    sorted_means = sorted(means)
    lower_ci_index = int(alpha/2.0*n_resample)
    upper_ci_index = int((1-alpha/2.0)*n_resample)
    return np.mean(data), (sorted_means[lower_ci_index], sorted_means[upper_ci_index])

def plot_pvalues_by_subject_with_confidence_intervals(pvalue_by_subject_dict, num_to_plot=10):
    plt.figure(figsize=(8,8))
    bootstrap_results = []
    for subject, pvalues in pvalue_by_subject_dict.items():
        mean, (lower_ci, upper_ci) = bootstrap_confidence_intervals_for_mean(pvalues)
        bootstrap_results.append((subject, mean, (lower_ci, upper_ci)))

    sorted_results = sorted(bootstrap_results, key=lambda x: x[1])

    subjects = [x[0] for x in sorted_results]
    lower_cis = [np.abs(x[1]-x[2][0]) for x in sorted_results]
    upper_cis = [np.abs(x[1]-x[2][1]) for x in sorted_results]
    means = [x[1] for x in sorted_results]

    if num_to_plot > 0:
        plt.barh(subjects[:num_to_plot], means[:num_to_plot], xerr=[lower_cis[:num_to_plot], upper_cis[:num_to_plot]])
    else:
        plt.barh(subjects[num_to_plot:], means[num_to_plot:], xerr=[lower_cis[num_to_plot:], upper_cis[num_to_plot:]])
    plt.xlim([0,1])
    plt.show()