"
Paper version (final)
"

constraints_ind <- read.csv('data/toy_constraints_ind3.csv', stringsAsFactors = F)

constraints_bg <- read.csv('data/toy_constraints_bg3.csv', stringsAsFactors = F)

trt_id <- substr(constraints_bg$GEOID, 1, 1)

est_cols <- c('POP', 'CONST1', 'CONST2', 'CONST3', 'CONST4', 'CONST5', 'CONST6')
se_cols <- paste0(est_cols, 's')

est_trt <- aggregate(constraints_bg[,est_cols], by = list(trt_id), FUN = sum)

agg_se <- function(se){
  sqrt(sum(se^2))
}

est_trt <- aggregate(constraints_bg[,est_cols], by = list(trt_id), FUN = sum)
est_se <- aggregate(constraints_bg[,se_cols], by = list(trt_id), FUN = agg_se)

constraints_trt <- merge(est_trt, est_se, by = 'Group.1')
names(constraints_trt)[1] <- 'GEOID'

geo_lookup <- data.frame(bg = constraints_bg$GEOID, trt = trt_id)

#### prep P-MEDM inputs ####
library(PMEDMrcpp)

serial <- constraints_ind$SERIAL
wt <- constraints_ind$PERWT

## individual-level constraints
pX <- as.matrix(constraints_ind[,-c(1:2)])
pX <- list(drop0(pX), drop0(pX))

## Geographies
A1 <- do.call('rbind',lapply(unique(geo_lookup[,2]),function(g){
  nbt <- geo_lookup[geo_lookup[,2]==g,][,1]
  ifelse(geo_lookup[,1] %in% nbt,1,0)
}))

rownames(A1) <- unique(geo_lookup[,2])
colnames(A1) <- geo_lookup[,1]
A1 <- as(A1,'dgCMatrix')

A2 <- do.call('rbind',lapply(geo_lookup[,1],function(g){
  ifelse(geo_lookup[,1] %in% g,1,0)
}))
rownames(A2) <- geo_lookup[,1]
colnames(A2) <- geo_lookup[,1]
A2=as(A2,'dtCMatrix')

A <- list(A1,A2)

## Geographic constraints
Y <- list(constraints_trt[,est_cols],
          constraints_bg[,est_cols])

## Error variances in geographic constraints
V <- list(constraints_trt[,se_cols]^2,
          constraints_bg[,se_cols]^2)

## Prep solver inputs ##
N <- sum(Y[[1]]$POP) # pop size
n <- nrow(constraints_ind) # sample size

# Since we are optimizing probabilities p, rather than weights w
#     Normalize Y by N and V by n/N^2
Y_vec <- do.call(c, lapply(Y,function(x) as.vector(as.matrix(x))))/N
V_vec <- do.call(c, lapply(V,function(x) as.vector(as.matrix(x))))*n/N^2

# Will need a matrix V, not a vector V
sV <- .sparseDiagonal(n=length(V_vec), x=V_vec);

# Create X matrix
X <- t(rbind(kronecker(t(pX[[1]]), A[[1]]), kronecker(t(pX[[2]]), A[[2]])))

# Create design weights and normalize
q <- matrix(wt, n, dim(A[[1]])[2])
q <- q/sum(q)
q <- as.vector(t(q))

X <- as(X, 'dgCMatrix')
sV <- as(sV, 'dgCMatrix')

t <- PMEDMrcpp::PMEDM_solve(X, Y_vec, sV, q) # test 

source('../pmedmize/code/reliability.R')
p.hat <- compute_allocation(q, X, t$lambda) # test
sum(t$p - p.hat)

#### Homebrew optimization ####

get_parent_ids <- function(XW){
  
  apply(XW, 2, function(x) rownames(XW)[which(x == 1)])
  
}

agg2parent <- function(pmat, ids){
  
  t(aggregate(t(pmat) ~ ids, FUN = sum)[,-1])
  
}

### Optimization ####
f <- function(lambda){
  
  qXl <- q * exp(-X %*% lambda)
  
  lvl = lambda %*% (sV %*% lambda)
  
  as.numeric(Y_vec %*% lambda + log(sum(qXl)) + (0.5 * lvl))
  
}

# pmedm.simple <- optim(par = rep(0, length(Y_vec)),
#                       fn = f,
#                       method = 'BFGS',
#                       control = list(trace = 4, maxit = 1000))

# # # compare P-MEDM full and P-MEDM simple coefficients
# # plot(pmedm.simple$par ~ t$lambda)
# 
## allocation with p-medm simple coefficients
p.hat <- compute_allocation(q, X, lambda = rep(0, length(Y_vec))) # init lambda

p.hat <- reshape_probabilities(p.hat, n, A)
p.hat.trt <- agg2parent(p.hat, ids = get_parent_ids(A[[1]]))

Y.hat <- c(
  as.vector(apply(pX[[1]], 2, function(v) colSums(v * p.hat.trt * N))),
  as.vector(apply(pX[[1]], 2, function(v) colSums(v * p.hat * N)))
)

Ype <- data.frame(Y = Y_vec * N, Y.hat = Y.hat, V = diag(sV) * N^2/n)
Ype$MOE.lower <- Ype$Y - (1.645 * sqrt(Ype$V))
Ype$MOE.upper <- Ype$Y + (1.645 * sqrt(Ype$V))
Ype$win.moe <- factor(with(Ype, ifelse(Y.hat >= MOE.lower & Y.hat <= MOE.upper, 'Yes', 'No')),
                      levels = c('No', 'Yes'))
table(Ype$win.moe) / nrow(Ype)

library(ggplot2)
plt.init <- ggplot(data = Ype, aes(x = Y, y = Y.hat)) +
  geom_linerange(aes(ymin = MOE.lower, ymax = MOE.upper, col = win.moe), size = 2, position = position_dodge2(width = 1)) +
  geom_point(aes(fill = win.moe), pch = 21, position = position_dodge2(width = 1)) +
  scale_color_manual(values = c('coral', 'skyblue'), drop = F) +
  scale_fill_manual(values = c('coral', 'skyblue'), drop = F) +
  ggtitle('Initial Solution') +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(color = 'Within ACS 90%\nMargin of Error?') + 
  guides(fill = 'none')

###
p.hat.optim <- compute_allocation(q, X, lambda = t$lambda) # P-MEDMrcpp final lambdas
p.hat.optim <- compute_allocation(q, X, lambda = pmedm.simple$par) # P-MEDM simple final lambda

p.hat.optim <- reshape_probabilities(p.hat.optim, n, A)
p.hat.trt.optim <- agg2parent(p.hat.optim, ids = get_parent_ids(A[[1]]))

Y.hat.optim <- c(
  as.vector(apply(pX[[1]], 2, function(v) colSums(v * p.hat.trt.optim * N))),
  as.vector(apply(pX[[1]], 2, function(v) colSums(v * p.hat.optim * N)))
)

Ype.optim <- data.frame(Y = Y_vec * N, Y.hat = Y.hat.optim, V = diag(sV) * N^2/n)
Ype.optim$MOE.lower <- Ype.optim$Y - (1.645 * sqrt(Ype.optim$V))
Ype.optim$MOE.upper <- Ype.optim$Y + (1.645 * sqrt(Ype.optim$V))
Ype.optim$win.moe <- factor(with(Ype.optim, ifelse(Y.hat >= MOE.lower & Y.hat <= MOE.upper, 'Yes', 'No')),
                            levels = c('No', 'Yes'))

plt.optim <- ggplot(data = Ype.optim, aes(x = Y, y = Y.hat)) +
  geom_linerange(aes(ymin = MOE.lower, ymax = MOE.upper, col = win.moe), size = 2, position = position_dodge2(width = 1)) +
  geom_point(aes(fill = win.moe), pch = 21, position = position_dodge2(width = 1)) +
  scale_color_manual(values = c('coral', 'skyblue'), drop = F) +
  scale_fill_manual(values = c('coral', 'skyblue'), drop = F) +
  ggtitle('Optimized Solution') +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(color = 'Within ACS 90%\nMargin of Error?') + 
  guides(fill = 'none')

cowplot::plot_grid(plt.init, plt.optim, nrow = 1)
