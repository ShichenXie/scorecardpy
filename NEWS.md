# scorecardpy 0.1.6.4

* pdo in scorecard function now suports negative value. If pdo is positive, the larger score means the lower probability to be positive sample. If pdo is negative, the larger score means the higher probability to be positive sample.
* fixed a bug in woebin function using chimerge method, which is caused by initial breaks have out-range values.
* added a check function on the length of unique values in string columns, which might cause the binning process slow.
* fixed a bug in perf_eva function which is caused by the nrow of plot is setted to 0 when the length of plot type is one.
* the ratio argument in split_df function supports to set ratios for both train and test.
* If the argument return_rm_reason is TRUE in var_filter function, the info_value, missing_rate and identical_rate are provided.

# scorecardpy 0.1.6

* fixed a bug in iv function if the variable is categorical
* fixed a bug in woebin function when specifing breaks omiting missing
* remove sort=False in concat, since it is only available in pandas 0.23
* fixed a bug in scorecard function, which is caused by unsorted dictionary in python below 3.6 version 
* fixed a bug in woebin function when breaks_list if provided for numeric variables causing unsorted bin.

# scorecardpy 0.1.5

* fixed a bug in woebin_ply function when spl_val is specified.
* fixed a bug in woebin function when there are NaNs in input dataframe.
* fixed a bug in perf_eva when calculating auc.
* fixed a bug in woebin_adj when special_values is provided.
* fixed a bug in iv function which is caused by pandas' groupby ignoring nan values.

# scorecardpy 0.1.4

* fixed a bug in var_filter function when the type of var_rm/var_kp is str
* fixed a bug in scorecard_ply function calculates the total score double. 
* fixed a bug in woebin function when breaks_list is provided for numeric variables.

# scorecardpy 0.1.3

* fixed the bug in woebin function when there are variables have only one unique value. 
* modify the default values of x_limits in perf_psi
* display proc time in woebin




# scorecard 0.1.7

* added chimerge method for woe binning

# scorecard 0.1.6

* Fixed a bug in woebin_adj function when all_var == FALSE and the breaks of all variables are perfect. 
* Provide parallel computation (foreach with parallel backend) in the functions of woebin and woebin_ply.
* Modified scorecard_ply function.
* Fixed a bug in woebin when there are empty bins based on provided break points. 

# scorecard 0.1.5

* Fixed a bug in scorecard function when calculating the coefficients.
* Fixed a bug in perf_eva when type="lift". 
* Fixed a bug in functions of woebin and var_filter when removing Date columns. 

# scorecard 0.1.4

* perf_eva supports both predicted probability and score.
* Added the woebin_adj function which can interactively adjust the binning info from woebin.
* Reviewed woebin function.

# scorecard 0.1.3

* Modified the format of printing message and added condition functions.
* Added the split_df function which split a dataframe into two.
* Reorder the binning information. Move the missing to the first binning.

# scorecard 0.1.2

* fixed a bug in var_filter

# scorecard 0.1.1

* Specified some potential problems via conditions
* Modified examples for most functions

# scorecard 0.1.0

* Initial version



