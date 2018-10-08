# ===========================================================================
# Code adapted from https://github.com/sanjaybasu/sprint-challenge.
# ===========================================================================

# ==== load relevant libraries

library(Hmisc)
library(matrixStats)

# ==== load relevant data

load("accord_cut.RData")
attach(accord_set)

cvd = (accord_set$censor_nmi==0)|(accord_set$censor_nst==0)|(accord_set$censor_cm==0)|(accord_set$censor_chf==0)|(accord_set$censor_maj==0)
t_censor = rowMaxs(cbind(accord_set$fuyrs_nmi*365.25,accord_set$fuyrs_nst*365.25,accord_set$fuyrs_cm*365.25,accord_set$fuyrs_chf*365.25,accord_set$fuyrs_maj*365.25))
t_cvds = rowMaxs(cbind(accord_set$fuyrs_nmi*365.25*(1-accord_set$censor_nmi),accord_set$fuyrs_nst*365.25*(1-accord_set$censor_nst),accord_set$fuyrs_cm*365.25*(1-accord_set$censor_cm),accord_set$fuyrs_chf*365.25*(1-accord_set$censor_chf),accord_set$fuyrs_maj*365.25*(1-accord_set$censor_maj)))
t_cvds[t_cvds==0] = t_censor[t_cvds==0]
t_cvds[t_cvds==0] = 'NA'
t_cvds = as.numeric(t_cvds)
cOutcome = Surv(time=t_cvds, event = cvd)
sae = (accord_set$advexp.max==1)
accord_set$visadv.min=as.numeric(accord_set$visadv.min)
t_censor = rowMaxs(cbind(accord_set$visnum.max*30.42))
t_saes = rowMaxs(cbind(accord_set$visadv.min*30.42))
t_saes[is.na(t_saes)] = t_censor[is.na(t_saes)]
t_saes[t_saes==0] = 'NA'
t_saes = as.numeric(t_saes)
dOutcome = Surv(time=t_saes, event = sae)
INTENSIVE = as.numeric(accord_set$INTENSIVE)
AGE = accord_set$baseline_age
FEMALE = accord_set$female
RACE_BLACK = as.numeric(accord_set$raceclass=="Black")
hisp = (accord_set$raceclass=="Hispanic")
SBP.y = accord_set$sbp
DBP.y = accord_set$dbp
N_AGENTS = (accord_set$loop+accord_set$thiazide+accord_set$ksparing+accord_set$a2rb+accord_set$acei+accord_set$dhp_ccb+accord_set$nondhp_ccb+accord_set$alpha_blocker+accord_set$central_agent+accord_set$beta_blocker+accord_set$vasodilator+accord_set$reserpine+accord_set$other_bpmed)
currentsmoker = (accord_set$cigarett==1)
formersmoker = (accord_set$smokelif==1)
ASPIRIN= accord_set$aspirin
STATIN = accord_set$statin
SUB_SENIOR = as.numeric(AGE>=75)
SUB_CKD = as.numeric(accord_set$gfr<60)
CHR = accord_set$chol
GLUR = accord_set$fpg
HDL = accord_set$hdl
TRR = accord_set$trig
UMALCR = accord_set$uacr
EGFR = accord_set$gfr
SCREAT = accord_set$screat
BMI = accord_set$wt_kg/((accord_set$ht_cm/1000)^2)/100
c2<-data.frame(cvd,t_cvds,
              INTENSIVE,AGE,FEMALE,RACE_BLACK,hisp,
              SBP.y,DBP.y,N_AGENTS,currentsmoker,formersmoker,
              ASPIRIN,STATIN,
              SCREAT,CHR,HDL,TRR,BMI)
c2=c2[complete.cases(c2),]
write.csv(c2, "./accord_cut.csv")