constraints_bg <- read.csv('data/toy_constraints_bg.csv', stringsAsFactors = F)

trt_id <- substr(constraints_bg$GEOID, 1, 2)

est_cols <- c('POP', 'CONST1', 'CONST2', 'CONST3')
se_cols <- paste0(est_cols, 's')

est_trt <- aggregate(constraints_bg[,est_cols], by = list(trt_id), FUN = sum)

agg_se <- function(se){
  sqrt(sum(se^2))
}

est_trt <- aggregate(constraints_bg[,est_cols], by = list(trt_id), FUN = sum)
est_se <- aggregate(constraints_bg[,se_cols], by = list(trt_id), FUN = agg_se)

constraints_trt <- merge(est_trt, est_se, by = 'Group.1')
names(constraints_trt)[1] <- 'GEOID'

write.csv(constraints_trt, file = 'data/toy_constraints_trt.csv')