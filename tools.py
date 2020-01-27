import numpy as np
import csv


def convert(file, delimiter="\t", comment_sign="#", columns=2, ignore_lines=1):
	data = []

	for _ in range(columns):
		data.append([])

	with open(file, "r") as csvfile:
		reader = csv.reader(csvfile, delimiter=delimiter)

		count = 0
		for row in reader:
			count = count + 1

			if count > ignore_lines:
				if not row[0].startswith(comment_sign):
					for i in range(columns):
						data[i].append(row[i])

	data_temp = []

	for i in range(len(data)):
		data_temp.append([])

		for j in range(len(data[i])):
			data_temp[i].append(float(data[i][j]))

	data = np.array(data_temp)

	if len(data) == 1:
		data = data[0]

	return data


def mean_with_error(data, print_res=False):
	data = np.array(data)

	# Mittelwert-Fehler:
	n = len(data)
	mittelwert = data.mean()

	o = 0
	for i in range(n):
		o += (data[i] - mittelwert)**2

	err = np.sqrt(o / (n * (n - 1)))

	if print_res:
		print(f"{mittelwert} +/- {err}")

	return mittelwert, err


def linfit(X, Y):
	"""
	Calculate a "MODEL-1" least squares fit.

	The line is fit by MINIMIZING the residuals in Y only.

	The equation of the line is:	 Y = my * X + by.

	Equations are from Bevington & Robinson (1992)
	Data Reduction and Error Analysis for the Physical Sciences, 2nd Ed."
	pp: 104, 108-109, 199.

	Data are input and output as follows:

	my, by, ry, smy, sby = linfit(X,Y)
	X	=	x data (vector)
	Y	=	y data (vector)
	my	=	slope
	by	=	y-intercept
	ry	=	correlation coefficient
	smy	=	standard deviation of the slope
	sby	=	standard deviation of the y-intercept

	"""

	X, Y = map(np.asanyarray, (X, Y))

	# Determine the size of the vector.
	n = len(X)

	# Calculate the sums.

	Sx = np.sum(X)
	Sy = np.sum(Y)
	Sx2 = np.sum(X ** 2)
	Sxy = np.sum(X * Y)
	Sy2 = np.sum(Y ** 2)

	# Calculate re-used expressions.
	num = n * Sxy - Sx * Sy
	den = n * Sx2 - Sx ** 2

	# Calculate my, by, ry, s2, smy and sby.
	my = num / den
	by = (Sx2 * Sy - Sx * Sxy) / den
	ry = num / (np.sqrt(den) * np.sqrt(n * Sy2 - Sy ** 2))

	diff = Y - by - my * X

	s2 = np.sum(diff * diff) / (n - 2)
	smy = np.sqrt(n * s2 / den)
	sby = np.sqrt(Sx2 * s2 / den)

	return my, by, ry, smy, sby


def table(content, header=[], position="", label="tab:my_label", caption="Caption"):
	# the content parameter is organized in columns
	content = col_to_row(content)
	content = [header] + content

	layout = "|c||"
	for _ in content[0][1:]:
		layout = layout + "c|"

	res = f"\\begin{{table}}[{position}]\n\t\\centering\n\t\\begin{{tabular}}{{{layout}}}\n\t\t\\hline\n\t\t"

	for obj in content[0]:
		res = res + str(obj) + " & "

	res = res[:-2] + "\\\\ \\hline\\hline\n\t\t"

	for row in content[1:]:
		for obj in row:
			res = res + str(obj) + " & "

		res = res[:-2] + "\\\\ \\hline\n\t\t"

	res = res[:-1] + f"\\end{{tabular}}\n\t\\caption{{{caption}}}\n\t\\label{{{label}}}\n\\end{{table}}"

	return res


def col_to_row(array):
	res = []

	for i in range(len(array[0])):
		row = []

		for j in range(len(array)):
			row.append(array[j][i])

		res.append(row)

	return res


def pm(data, error):
	res = []
	try:
		for i in range(len(data)):
			res.append("$" + str(data[i]) + " \\pm " + str(error[i]) + "$")
	except TypeError:
		res = "$" + str(data) + " \\pm " + str(error) + "$"

	return res


def plt_linfit(x, y, plt, label="linearer Fit"):
	m, b, r, m_err, b_err = linfit(x, y)

	x_fit = np.linspace(np.min(x), np.max(x), 256)
	y_fit = m * x_fit + b

	plt.plot(x_fit, y_fit, label=label)

	return m, b, m_err, b_err


def weighed_mean(data, errors):
	data = np.array(data)
	errors = np.array(data)

	mean = np.sum(data / errors**2) / np.sum(1 / errors**2)
	error = np.sqrt(1 / np.sum(1 / errors**2))
	error_extended = np.sqrt(np.sum((data - mean)**2 / errors**2) / ((len(data) - 1) * np.sum(1 / errors**2)))

	return mean, error, error_extended