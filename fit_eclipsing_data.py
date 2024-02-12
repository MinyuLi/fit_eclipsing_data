import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FixedLocator
from multiprocessing import Pool
import os
import time as tm
import corner
import emcee
from scipy.stats import gaussian_kde


rcParams["savefig.dpi"] = 300
rcParams['font.size'] = 8

result_path=os.path.join('.')
label = ['$\mathit{a}$', '$\mathit{T_{0sc}}$', '$\mathit{P}$', '$\mathit{c}$']
ndim = 4
nwalkers = 50
nsteps = 10000

target_name='KIC 7914906'
src_file = '07914906.00.lc.data'
P_est = 8.7529074 #轨道周期预估值 
T0sc_est = 54962.971405 #T0sc预估值



def get_data_fullpath(fullpath, outliner=False):
	try:
		time, flux, flux_err = np.loadtxt(fullpath, usecols=(0,6,7), dtype="float", unpack=True, comments='#', ndmin=2)
	except Exception as e:
		time, flux, flux_err = np.loadtxt(fullpath, usecols=(0,1,2), dtype="float", unpack=True, comments='#', ndmin=2)
		return time, flux, flux_err
	lc = lk.LightCurve(time=time, flux = flux, flux_err = flux_err)
	lc = lc.normalize()
	if outliner:
		lc = lc.remove_outliers()
	time = lc.time.value
	flux_norm = lc.flux
	flux_err_norm = lc.flux_err
	return time, flux_norm, flux_err_norm


def get_eclipsing_data(file_path, P, T0, phase_min, phase_max, flux_max):
	#本例中 Kirk 数据在第6，7列。需根据实际情况调整
	x, y, yerr = get_data_fullpath(file_path)

	phase_0_1=(x-T0)%P/P #x转换到 [0, 1]相位
	phase_x = (phase_0_1 >= 0.5).astype(int)*(phase_0_1-1.) + (phase_0_1 < 0.5).astype(int)*phase_0_1 #[0, 1]相位转换到[-0.5, 0.5]相位

	L1 = (phase_x > phase_min) & (phase_x < phase_max)
	L2 = y < flux_max
	x = np.array(x[L1&L2])
	y = np.array(y[L1&L2])
	yerr = np.array(yerr[L1&L2])

	return x, y, yerr

def model(x, theta):
	a,T0,P,c = theta
	phase_0_1=np.array((x-T0)%P/P) #x转换到 [0, 1]相位
	phase_x = (phase_0_1 >= 0.5).astype(int)*(phase_0_1-1.) + (phase_0_1 < 0.5).astype(int)*phase_0_1 #[0, 1]相位转换到[-0.5, 0.5]相位

	return a*phase_x**2+c


def log_likelihood (x,y,er, theta):
	sigma2 = er ** 2
	return -0.5*np.sum( (y-model(x,theta))**2/sigma2+np.log(2*np.pi*sigma2))


def log_prior_p(theta):
	a,b,c,d =theta
	if (0. < a < 10000.):
		return 0

	return -np.inf

def log_probability(theta,x,y,yerr):
	lp=log_prior_p(theta)
	if not np.isfinite(lp):
		return -np.inf
	try:
		likely = log_likelihood(x,y,yerr,theta)
		if not np.isfinite(likely):
			return -np.inf
		return lp+likely
	except Exception as e:
		print(f"log_likelihood exception. theta={theta}")
	return -np.inf

def show_fit_proc(sampler, show = True, save = True, round=0, no=0):
	fig, axes = plt.subplots(ndim, figsize=(10, 5), sharex=True)
	samples = sampler.get_chain()
	for i in range(ndim):
		ax = axes[i]  if ndim > 1 else axes
		s = samples[:, :, i]
		ax.plot(s, "k", alpha=0.3)
		ax.set_xlim(0, len(samples))
		ax.set_ylabel(label[i])
		ax.yaxis.set_label_coords(-0.1, 0.5)

	ax.set_xlabel("step number")
	plt.suptitle(target_name, fontsize=10)

	if save:
		filepath = os.path.join(result_path, target_name+"-FitProc"+"-%03d" % round+"-%03d" % no+".png")
		plt.savefig(filepath)
	if show:
		plt.show()
	plt.close()

def show_fit_par_hitogram(sampler, show = True, save = True, round=0, no=0):
	try:
		tau = sampler.get_autocorr_time()
		burnin = int(2 * np.max(tau))
		thin = int(0.5 * np.min(tau))
		samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
		log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
		log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)
	except Exception as e:
		burnin = int(nsteps/10)
		thin = int(ndim)

	fig, axes = plt.subplots(1,ndim, figsize=(10, 5), sharey=True, tight_layout=True)
	flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
	par1 = []
	pos = 0
	for i in range(ndim):
		ax = axes[pos] if ndim > 1 else axes
		pos += 1
		ax.hist(flat_samples[:, i], 10)
		ax.set_ylabel(label[i])
		ax.yaxis.set_label_coords(-0.1, 0.5)

	ax.set_xlabel("percents")
	plt.suptitle(target_name, fontsize=10)

	if save:
		filepath = os.path.join(result_path, target_name+"-ParHitogram"+"-%03d" % round+"-%03d" % no+".png")
		plt.savefig(filepath)
	if show:
		plt.show()
	plt.close()

def get_likelihood_var(sampler, total_runned_step=None):
	autocorr = False
	try:
		tau = sampler.get_autocorr_time()
		burnin = int(2 * np.max(tau))
		thin = int(0.5 * np.min(tau))
		samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
		log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
		log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)

		print("burn-in: {0}-{1}, burnin={2}".format(target_name, 0, burnin) +\
					"thin: {0}".format(thin))
		autocorr = True
	except Exception as e:
		if total_runned_step is None:
			burnin = int(nsteps/10)
		else:
			burnin = int(total_runned_step/10)
		thin = int(ndim)
		errmsg = f"get_autocorr_time except{target_name}:{str(e)}.\nset burnin={burnin}, thin={thin}"
		print(errmsg)

	flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
	params=[]
	err1=[]
	err2=[]
	for pos in range(ndim):
		mcmc = np.percentile(flat_samples[:, pos], [16, 50, 84])
		q = np.diff(mcmc)
		params.append(mcmc[1])
		err1.append(-q[0])
		err2.append(q[1])

	return autocorr, params, err1, err2, flat_samples

def show_corner(flat_samples, truths,      show=True, save=True, round=0, no=0):
	#flat_samples = self.sampler.get_chain(flat=True, thin=self.ndim)
	f = rcParams['font.size']
	rcParams['font.size'] = 5
	
	fig = plt.figure(figsize=(10, 10))
	fig2 = corner.corner(flat_samples, labels=label, hist_kwargs={'density': True},
			truths=truths,
			fig =fig, truth_color='firebrick', 
			#plot_datapoints=False, bins=20, 
			quantiles=[0.16, 0.5, 0.84], max_n_ticks=3,
			show_titles=True, title_fmt=".4f", title_kwargs={"fontsize": 7}, titles=label,
			label_kwargs={"fontsize": 10})
	#fig2.set_alpha(0.3)
	pos = [i*(len(label)+1) for i in range(len(label))]
	for axidx, samps in zip(pos, flat_samples.T):
		kde = gaussian_kde(samps)
		xvals = fig.axes[axidx].get_xlim()
		xvals = np.linspace(xvals[0], xvals[1], 50)
		fig.axes[axidx].plot(xvals, kde(xvals), color='firebrick')

	if save:
		filepath = os.path.join(result_path, target_name + "-Corner"+"-%03d" % round+"-%03d" % no+".png")
		savedpi = rcParams["savefig.dpi"]
		rcParams["savefig.dpi"] = 360
		plt.savefig(filepath)
		rcParams["savefig.dpi"] = savedpi
	if show:
		plt.show()
	plt.close()
	rcParams['font.size'] = f

def show_fit_result(x0,y,params, show=True, save=True, round=0, no=0):
	a,T0,P,c =params
	phase_0_1=np.array((x0-T0)%P/P) #x转换到 [0, 1]相位
	x = (phase_0_1 >= 0.5).astype(int)*(phase_0_1-1.) + (phase_0_1 < 0.5).astype(int)*phase_0_1 #[0, 1]相位转换到[-0.5, 0.5]相位
	#if prefix == 0:
	#	a = 7.78813949
	#	b = -2*a*54954.231104
	#	c = a*54954.231104**2 + 0.96330661
	plot_len = 4
	xcale1 = np.min(x)-2*np.std(x)
	xcale2 = np.max(x)+2*np.std(x)
	
	#for phoebe fittint results
	odder_pos = np.argsort(x)
	x_line = np.array(x[odder_pos])
	y_line = a*x_line**2+c

	fitflux = model(x0, params)
	deltaflux = y - fitflux 
	line_width = 1

	fig, axs = plt.subplot_mosaic([
		    ['signal'],['signal'],['signal'], #set ratio of the two view as 3:1
		    ['residual'],
		], layout='constrained', sharex=True, figsize=(plot_len,5))
	ax1 = axs['signal']
	ax2 = axs['residual']

	alpha=0.8
	#ax1.plot( phase_+1., fitflux*1000., "r", ms=1, lw=line_width)
	ax1.scatter(x, y, c="k", s=1, alpha=alpha)
	ax1.plot(x_line, y_line, "r", ms=1, lw=line_width)
	ax1.set_ylabel("Normalized flux", fontsize='large')
	ax1.set_xlim(xcale1, xcale2)
	#ax1.xaxis.set_major_locator(MultipleLocator(0.25))
	#ax1.set_ylim(ylim_low, ylim_high)

	# plot zoomed in view of transits
	color = "k"
	ax2.scatter(x, deltaflux, c="k", s=1, alpha=alpha)

	ax2.plot([-2.,2.], [0.,0.], "r--", ms=1, lw=1)
	ax2.set_ylabel("Residuals", fontsize='large')
	ax2.set_xlabel("Orbital phase", fontsize='large')
	ax2.set_xlim(xcale1, xcale2)
	ax2.set_ylim(-0.011, 0.01)
	ax2.xaxis.set_major_locator(MultipleLocator(0.01))

	title = target_name + "\n$\mathit{T_{0sc}}$" + "={0:.6f}".format(T0) + "BJD(-2400000), " +\
				"$\mathit{P}$" + "={0:.7f} d".format(P)
			
	plt.suptitle(title, fontsize=10)
	if save:
		filepath = os.path.join(result_path, target_name + "-FitResult"+"-%03d" % round+"-%03d" % no+".png")
		plt.savefig(filepath)
	if show:
		plt.show()
	plt.close()

def save_fit_results(x,y,params, err1, err2):
	a,T0,P,c = params
	rst = "{0:s} ".format(target_name) + \
		"{0:.8f} ".format((np.sum((y-model(x, params))**2)/len(x))**1) + \
		"{0:.8f} ".format(a) + "{0:.8f} ".format(err1[0]) + "{0:.8f} ".format(err2[0]) + \
		"{0:.8f} ".format(T0) + "{0:.8f} ".format(err1[1]) + "{0:.8f} ".format(err2[1]) + \
		"{0:.8f} ".format(P) + "{0:.8f} ".format(err1[2]) + "{0:.8f} ".format(err2[2]) + \
		"{0:.8f} ".format(c) + "{0:.8f} ".format(err1[3]) + "{0:.8f} ".format(err2[3]) + \
		"\n"
	rstfile =  os.path.join(result_path, "O_C_rst.dat") #The results are collected in one file
	with open(rstfile, 'a') as fw:
		try:
			fw.write(rst)
		except Exception as e:
			print(f"{target_name} write file fail. except:{str(e)}")


def mcmc_run(x,y,yerr,pos,round):
	each_step=2000
	with Pool() as pool:
		sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, args=(x,y,yerr))
		samples=pos
		no = 0
		finished_steps = 0
		while finished_steps < nsteps:
			samples = sampler.run_mcmc(samples, each_step, progress=True)
			finished_steps = finished_steps + each_step
			

			show_fit_proc(sampler, show=False, round=round, no=no)
			show_fit_par_hitogram(sampler, show=False, round=round, no=no)
			autocorr, params, err1, err2, flat_samples = get_likelihood_var(sampler, total_runned_step=finished_steps)
			show_corner(flat_samples,params, show=False, round=round, no=no)
			show_fit_result(x,y,params, show=False, round=round, no=no)
			#save_fit_results(params, err1, err2, prefix)
			if autocorr == True:
				print(f"autocorr, stop {no}.")
				break
			no = no+1
		save_fit_results(x,y,params, err1, err2)
		return params, autocorr



if __name__ == "__main__":
	x,y,yerr = get_eclipsing_data(src_file, P_est, T0sc_est, -0.01, 0.01, 0.99)
	pos=[]
	pos.append(np.random.uniform(750,850,nwalkers))
	pos.append(np.random.uniform(T0sc_est-0.1,T0sc_est+0.1,nwalkers))
	pos.append(np.random.uniform(P_est-0.1,P_est+0.1,nwalkers))
	pos.append(np.random.uniform(0.9,1,nwalkers))
	pos = np.array(pos).T
	params, _=mcmc_run(x, y, yerr, pos, 1)
	print( params)

	pos=[]
	pos.append(np.random.normal(params[0],np.abs(0.001),nwalkers))
	pos.append(np.random.normal(params[1],np.abs(0.001),nwalkers))
	pos.append(np.random.normal(params[2],np.abs(0.001),nwalkers))
	pos.append(np.random.normal(params[3],np.abs(0.001),nwalkers))
	pos = np.array(pos).T
	params, _=mcmc_run(x, y, yerr, pos, 2)
	print( params)
	
	
