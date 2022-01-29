# # installs
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
mydata = read.csv("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/src/assets/df_scores.csv")

group <- rep(rep(c("A", "A", "B", "A", "A", "B", "B", "B", "A", "B"),each=2), 3)
mydata["group"] = group
head(mydata)

# make columns factors
mydata$Stim <- as.factor(mydata$Stim)
mydata$Period <- as.factor(mydata$Period)
mydata$Task <- factor(ifelse(mydata$Task == 1,"Pre","Post"), levels = c("Pre", "Post"))
mydata$ID <- as.factor(mydata$ID)
str(mydata)

# split data based on nback
d1 = mydata %>% filter(Nback == 1)
d2 = mydata %>% filter(Nback == 2)
d3 = mydata %>% filter(Nback == 3)

# Set up 
m <- lmer(Score ~ Stim*Period + (1|ID), data=d3)
summary(m)
anova(m)
step(m, reduce.fixed=FALSE, reduce.random = TRUE)

#m1 <- lm(Score ~  Stim, data=mydata)
#m2 <- lmer(Score ~ -1 + Stim:group + (1 | ID) + (1 | Period), data=d2)

# Perform post-hoc test 
emm <- emmeans(m, "Stim")
pairs(emm)
summary(glht(m, linfct = mcp(Stim = "Tukey"), alternative = c("greater"), df=16), test = adjusted("fdr"))

# Plot a t-distribution
pt = dist_t(
  #t = NULL,
  #main = "title",
  deg.f = 18,
  p = 0.05,
  xmax = 3.5,
  #geom.colors = NULL,
  geom.alpha = 0.7
  #title="t-distribution",
)
pt.add_title("t-distribution")
plot(pt)

# plot model 
p <- plot_model(m)
p + theme_sjplot()

#
d3_stim0 = d1 %>% filter(d3$Stim == 0)
d3_stim1 = d1 %>% filter(d3$Stim == 1)

# Make theme
mytheme <- theme(text = element_text(size=20), 
        axis.title.y = element_text(angle=90),
        legend.position = c(0.25,0.1), legend.key = element_blank(), legend.direction = "horizontal",
        legend.background = element_rect(size=0.5, linetype = "solid", colour = "black"),
        plot.title = element_text(size=25, hjust=0.5, face="bold"))

# spagetti plot
p1 <- ggplot(data = d3_stim0, aes(x = Task, y = Score, group = ID, colour=Period)) +
  mytheme + 
  geom_line(size=1.5) + 
  ggtitle("Continuous") +
  coord_trans(ylim = c(73,100))  + 
  scale_color_manual(values=c("#007ea7", "#00a8e8")) +
  labs(y = "Percent Accuracy [%]", x = "Stimulation") + 
  scale_x_discrete(limits = c("Pre","Post"), expand = c(0.1, 0.1))
  
p2 <- ggplot(data = d3_stim1, aes(x = Task, y = Score, group = ID, colour=Period)) +
  mytheme + 
  geom_line(size=1.5) + 
  ggtitle("ISF") +
  labs(x = "Stimulation") + 
  coord_trans(ylim = c(73,100))  + 
  scale_color_manual(values=c("#007ea7", "#00a8e8")) +
  labs(y = "Percent Accuracy [%]", x = "Stimulation") + 
  scale_x_discrete(limits = c("Pre","Post"), expand = c(0.1, 0.1))
  #theme(axis.title.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank())

grid.arrange(p1,p2, ncol=2)
str(d3)

# sViolin
p1 <- ggplot(data = d3_stim0, aes(x = Task, y = Score, fill=Task)) +
  mytheme + 
  geom_violin(trim=FALSE) + 
  geom_boxplot(width=0.1, fill = "white") +
  ggtitle("Continuous") +
  coord_trans(ylim = c(60,110))  + 
  scale_color_manual(values=c("#007ea7", "#00a8e8")) +
  #geom_dotplot(binaxis='y', stackdir='center', position=position_dodge(1)) +
  labs(y = "Percent Accuracy [%]", x = "Stimulation") +
  scale_x_discrete(limits = c("Pre","Post"), expand = c(0.5, 0.1))

p2 <- ggplot(data = d3_stim1, aes(x = Task, y = Score, fill=Task)) +
  mytheme + 
  geom_violin(trim=FALSE) +
  geom_boxplot(width=0.1, fill = "white") +
  ggtitle("ISF") +
  labs(x = "Stimulation") + 
  #geom_dotplot(binaxis='y', stackdir='center', position=position_dodge(1)) +
  coord_trans(ylim = c(60,110))  + 
  scale_color_manual(values=c("#007ea7", "#00a8e8")) +
  labs(y = "Percent Accuracy [%]", x = "Stimulation") + 
  scale_x_discrete(limits = c("Pre","Post"), expand = c(0.5, 0.1))
  theme(axis.title.y = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank())

grid.arrange(p1,p2, ncol=2)

