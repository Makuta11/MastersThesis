# installs
# install.packages("lme4")
# install.packages("lmerTest")
# install.packages("dplyr")
# install.packages("emmeans")
# install.packages("pbkrtest")
# install.packages("multcomp")
# install.packages("nlme")
# install.packages("Crossover")
# install.packages("rJava")
# install.packages("plotly")
# install.packages("sjPlot")

# package imports
library(lme4)
library(ggplot2)
library(sjmisc)
library(sjPlot)
library(plotly)
library(rJava)
library(Crossover)
library(lmerTest)
library(dplyr)
library(emmeans)
library(pbkrtest)
library(multcomp)
library(gridExtra)
library(nlme)

# clear enviornment of variables
rm(list=ls())

# load data
mydata = read.csv("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EEG/assets/df_snr_stim")
write.csv(mydata, "/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EEG/assets/df_snr_stim.csv")
#group <- rep(rep(c("A", "A", "B", "A", "A", "B", "B", "B", "A", "B"), each=2), 3)
#mydata["group"] = group
head(mydata)

# make columns factors
mydata$Stimulus <- as.factor(mydata$Stimulus)
mydata$session <- as.factor(mydata$session)
mydata$ID <- as.factor(mydata$ID)
str(mydata)

# Set up 
m <- lme(SNR ~ Stimulus*session, random = ~1 | ID, data=mydata)
aov(m)
summary(m)
anova(m)

# Perform post-hoc test 
emm <- emmeans(m, "Stimulus")
emm <- emmeans(m, "session")
pairs(emm)

summary(glht(m, linfct = mcp(Stimulus = "Tukey"), alternative = c("greater"), df=16), test = adjusted("fdr"))

# plot model 
p <- plot_model(m)
p + theme_sjplot()

# Make theme
mytheme <- theme(text = element_text(size=20), 
                 axis.title.y = element_text(angle=90),
                 legend.position = c(0.5,0.5), legend.key = element_blank(),
                 legend.background = element_rect(size=0.5, linetype = "solid", colour = "black"),
                 plot.title = element_text(size=25, hjust=0.5, face="bold"))

# spagetti plot
p <- ggplot(data = mydata, aes(x = Stimulus, y = SNR, fill=Stimulus)) +
  mytheme + 
  geom_violin(trim=FALSE) + 
  geom_boxplot(width=0.1, fill="white") +
  ggtitle("Continuous") +
  #coord_trans(ylim = c(73,100))  + 
  guides(fill= guide_legend(reverse = TRUE)) +
  scale_color_manual(values=c("#007ea7", "#00a8e8",)) +
  labs(y = "Percent Accuracy [%]", x = "Stimulation") 
#scale_x_discrete(limits = c("Pre","Post"), expand = c(0.1, 0.1))


p



