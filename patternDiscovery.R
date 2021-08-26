# Huy Pham
# UC Berkeley
# Created: September 2020
# Description: This script plots data from the isolation data

library(ggplot2)
library(tidyverse)

rm(list = ls())

point.col = rgb(red=0.3, green=0.3, blue=0.3, alpha=0.4)  # point color for the plots
par(mfrow = c(2,2))

dataPath <- './imStudyDataFull.csv'
isol.full <- read.csv(dataPath, header=TRUE) %>% 
  filter(GMScale <= 20) %>% filter(GMSTfb <= 5)

isol.full$maxDrift <- pmax(isol.full$driftMax1, isol.full$driftMax2, isol.full$driftMax3)
isol.full$collapse <- ((isol.full$collapseDrift1 | isol.full$collapseDrift2) |
                         isol.full$collapseDrift3) %>% as.integer()
zetaRef     <- c(0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
BmRef       <- c(0.8, 1.0, 1.2, 1.5, 1.7, 1.9, 2.0)
tmp <- unlist(approx(zetaRef, BmRef, isol.full$zetaM)[2])
isol.full$Bm  <- tmp

g <- 386.4

# nondims
isol.full$TfbRatio <- isol.full$Tfb/isol.full$Tm
isol.full$mu2Ratio <- isol.full$mu2/(isol.full$GMSTm / isol.full$Bm)
# isol.full$gapRatio <- isol.full$moatGap/(isol.full$mu2 * g * isol.full$Tm^2)
isol.full$gapRatio <- isol.full$moatGap*4*pi^2/((isol.full$GMSTm/isol.full$Bm) *
                                                  g * isol.full$Tm^2)
isol.full$T2Ratio <- isol.full$T2/isol.full$Tm
isol.full$Qm <- isol.full$mu2*g

getDesignSa <- function(Tquery, S1) {
  Ss <- 2.2815
  Tshort <- S1/Ss
  if (Tquery < Tshort) {
    SaTquery <- S1
  } else {
    SaTquery <- S1/Tquery
  }
  return(SaTquery)
}

isol.full$S1Dm <- isol.full$moatGap*4*pi^2*isol.full$Bm/(g*isol.full$Tm)
isol.full$Sm <- mapply(getDesignSa, isol.full$Tm, isol.full$S1) * isol.full$Bm

isol.full$Pi1 <- isol.full$GMSavg/isol.full$S1Dm
isol.full$Pi2 <- isol.full$GMSavg/isol.full$Sm
isol.full$Pi3 <- isol.full$GMST2/isol.full$S1Dm
isol.full$Pi4 <- isol.full$GMST2/isol.full$Sm
isol.full$Pi5 <- isol.full$GMSTm/isol.full$S1Dm
isol.full$Pi6 <- isol.full$GMSTm/isol.full$Sm
isol.full$Pi7 <- isol.full$IPTm/(isol.full$S1Dm*g*isol.full$moatGap)
isol.full$Pi8 <- isol.full$PGA/(isol.full$S1Dm*g)
isol.full$Pi9 <- isol.full$PGV/(isol.full$S1Dm*g*isol.full$Tm)
isol.full$Pi10 <- isol.full$FIV3Tm/(isol.full$S1Dm*g*isol.full$Tm)

set.seed(1)

isol.train <- isol.full %>% sample_frac(0.8)
isol.test <- isol.full %>% setdiff(isol.train)

logiStudy <- function(piVar, train, test) {
  logitCollapse <- glm(paste("collapse ~ ", piVar), family=binomial(link = "logit"), 
                       data = train)
  summary(logitCollapse)
  confint(logitCollapse)
  test.prob <- logitCollapse %>% predict(test, type = "response")
  test.collapse <- ifelse(test.prob > 0.5, 1, 0)
  
  test.accuracy <- mean(test.collapse == test$collapse)
  
  return(list(classification = logitCollapse, accuracy = test.accuracy))
}

logiPlot <- function(dataSet = isol.train, mapping) {
    ggplot(data = dataSet, mapping) +
    geom_point(alpha = 0.2) +
    geom_smooth(method = "glm", method.args = list(family = "binomial")) +
    labs(
      title = "Logistic Regression Model", 
      x = "Pi",
      y = "Probability of collapse"
    )
}

dummy <- logiStudy("Pi1", isol.train, isol.test)
logiPlot(isol.train, aes(Pi1, collapse))
