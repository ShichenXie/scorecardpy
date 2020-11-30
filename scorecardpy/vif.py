import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif(df, y, x, merge_coef = False):
  import scorecardpy as sc
  df = sc.germancredit()
  y = 'creditability'
  x = ['age_in_years', 'credit_amount', 'present_residence_since']
  Xtrain = df.loc[:,x] 
  ytrain = df.loc[:,y] 
  Xtrain = sm.add_constant(Xtrain)

  lrfit = sm.GLM(
    ytrain.astype(float), 
    Xtrain.astype(float), 
    family=sm.families.Binomial()
  ).fit()

  y, X = dmatrices(' ~ '.join([y, '+'.join(x)]), data=df, return_type="dataframe")

  vif = pd.DataFrame({
      'variables': ['const', 'age_in_years', 'credit_amount', 'present_residence_since'], 
      'vif': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
  })

  if merge_coef:
    vif = pd.merge(
        lrfit.summary2().tables[1].reset_index().rename(columns = {'index':'variables'}), 
        vif,
        on = 'variables', how='outer'
    )
  
  return vif


    
#' Variance Inflation Factors
#'
#' \code{vif} calculates variance-inflation and generalized variance-inflation factors for linear, generalized linear.
#'
#' @param model A model object.
#' @param merge_coef Logical, whether to merge with coefficients of model summary matrix. Defaults to FALSE.
#'
#' @return A data frame with columns for variable and gvif, or additional columns for df and gvif^(1/(2*df)) if provided model uses factor variable.
#'
#' @seealso \url{https://cran.r-project.org/package=car}
#' @examples
#' data(germancredit)
#'
#' # Example I
#' fit1 = glm(creditability~ age.in.years + credit.amount +
#'   present.residence.since, family = binomial(), data = germancredit)
#' vif(fit1)
#' vif(fit1, merge_coef=TRUE)
#'
#' # Example II
#' fit2 = glm(creditability~ status.of.existing.checking.account +
#'   credit.history + credit.amount, family = binomial(), data = germancredit)
#' vif(fit2)
#' vif(fit2, merge_coef=TRUE)
#'
#'
#' @importFrom stats coef coefficients cov2cor model.matrix vcov
#' @export
vif = function(model, merge_coef = FALSE) {
  . = df = gvif = gvif_adj = variable = NULL

  if (any(is.na(coef(model)))) stop ("There are aliased coefficients in the model")

  v <- vcov(model)
  assign <- attr(model.matrix(model), "assign")
  if (names(coefficients(model)[1]) == "(Intercept)") {
    v <- v[-1, -1]
    assign <- assign[-1]
  } else warning("No intercept: vifs may not be sensible.")

  terms <- labels(terms(model))
  if (length(terms) < 2) stop("model contains fewer than 2 terms")

  R <- cov2cor(v)
  detR <- det(R)

  result <- data.table(variable=terms, gvif=0, df=0, gvif_adj=0) # generalized vif, degree freedom,
  for (t in seq_len(length(terms))) {
    subs = which(assign == t)
    result[t, `:=`(
      gvif = det(as.matrix(R[subs, subs])) * det(as.matrix(R[-subs, -subs])) / detR,
      df = length(subs) )]
  }
  if (result[, all(df==1)]) {
    result = result[,.(variable, gvif)]
  } else {
    result[, gvif_adj := gvif^(1/(2*df))]
    setnames(result, c('variable', 'gvif', 'df', 'gvif^(1/(2*df))'))
  }

  # merge with coefficients matrix
  if (merge_coef) {
    if (length(assign) == length(terms)) {
      coefDF = as.data.frame(coef(summary(model)))
      coefDT = data.table(variable = row.names(coefDF),Estimate=coefDF[,1],
                 data.table(coefDF[,2:4])[,lapply(.SD,function(x) round(x,4))])
      result = merge(coefDT, result, by='variable', all.x = TRUE, sort = FALSE)
    } else {
      warning('The summary matrix cant merge with vif.')
    }
  }

  return(result[])
}

# modified from car::vif
