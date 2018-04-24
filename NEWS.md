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



