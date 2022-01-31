### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 5f9b5d30-68cb-11ec-2e1e-aff540c73252
using SymbolicRegression, CSV, DataFrames, MultivariateStats, Statistics, Plots, HTTP, PlutoUI, SymbolicUtils, GLM

# ╔═╡ ecfd3b68-0f66-4695-87a3-404eafad37bc
md"""# Project: Genetic Programming for Regression

**STMO**

2021-2022

project by Jasper Staut"""

# ╔═╡ 94e88596-464d-41e7-a0da-68fc69f4e9c6
md"""## The concept
Apply the `SymbolicRegression` package for a regression problem and compare it to classical linear regression.

### The Data set
For this data set, I tracked my mood state each day over a period of 644 days. It includes following variables:

**Response variables** (score 0-10)
- frustration
- energy
- clarity
- happiness
- guilt
- emotionality
- anxiety
- confidence

**Predictor variables**
- sleepStart (bed time, time of day with 0 being midnight)
- sleepEnd (waking up, time of day with 0 being midnight)
- sleepDuration (amount slept, hours)
- sleepNap (napped during day, hours)
- meditation (score 0-2)
- exercise (physical exercise, score 0-2)

### The SymbolicRegression package
The `SymbolicRegression` package uses genetic programming to find a symbolic formula that predicts a response variable, using a set of predictor variables. Under the hood, the symbolic equations are represented as trees with nodes corresponding to an operator. The binary operator `+` for example can be linked to a node of a tree with two branches that respresent the two terms of the sum, each of which can be a sub-equation (or subtree) themselves. The leafs of these equation trees are either numbers or variables.

In `EquationSearch`, the central function of the package, these tree-equations are evolved every iteration into similar but better performing equations. A set of promising equations are kept in the *Hall of fame* and are evolved in further iterations.

Lastly, the dominating Pareto frontier is calculated for the equations in the hall of fame. This corresponds to the equations for which all simpler equations perform worse.

An advantage here is of course that non-linear relations can be modelled by including non-linear operators in your equations.
"""

# ╔═╡ d09fa8c4-c19d-4e5f-9d5f-c7831ec3762b
md"## Explore and prepare data set"

# ╔═╡ cc166935-f9a3-4b8e-b953-d97034e45dca
md"Read in data set"

# ╔═╡ 4ec5675e-a40b-4f12-bffc-9c6aeeebf7c0
dataset = DataFrame(CSV.File(HTTP.get("https://github.com/jstaut/GeneticProgrammingRegression.jl/raw/main/moodsData.csv").body; header=1, delim=";"));

# ╔═╡ ee67a2b5-8cb8-427f-9f38-286671c6b482
md"#### The response variables"

# ╔═╡ 5e762872-22a7-40aa-ae5a-441fdbcf462e
md"First, let's take a look the response variables:\
(note that this data set is a time series)"

# ╔═╡ 13af064b-9abc-4c8e-9ada-831910ced6fb
response = dataset[:,vcat(2:9)] # separate the response variables

# ╔═╡ b001aa84-527b-4ccd-96aa-8c4c9434870a
respNames = names(response);

# ╔═╡ 370172dc-15b3-490b-aa69-eb61d4272344
md"Let's look at a plot that visualizes the correlations between the variables"

# ╔═╡ 1b159d85-f073-4c0a-ab88-1ba7b54eae3b
heatmap(respNames, respNames, cor(Array(response)), color=:bluesreds, title="Correlation plot response variables")

# ╔═╡ 2872be01-1202-4cbc-8fe0-19da723b5848
md"It seems like most of these variables are correlated to each other, with the exception of `emotionality`"

# ╔═╡ 1d0c8ca3-6eeb-4fcb-b813-112895aa7103
md"Let's simplify the response variables by collapsing all variables except `emotionality` together into a single variable: `wellbeing`

PCA is used to do this:"

# ╔═╡ e5327245-9eed-44a8-af67-aca7160ac3a9
wellbeingOriginal = Matrix{Float64}(dataset[:,vcat(2:6,8:9)])';

# ╔═╡ 8ed46b21-3cd3-4f95-956f-0c1498273e11
M = fit(PCA, wellbeingOriginal; maxoutdim=1);

# ╔═╡ 43a4f4a1-2fa3-4eaf-bbae-7c484d0a6ee0
wellbeing = MultivariateStats.transform(M, wellbeingOriginal)'[:,1];

# ╔═╡ 82a567c0-ee21-4e5c-a079-736cf5a9493a
md"The `wellbeing` variable explains $(round(principalratio(M)*100)) % of the variance of all 7 original variables. We conclude that `wellbeing` is a decent representation of the original variables."

# ╔═╡ 8e9f1328-8a8e-4629-8004-8efd31429b69
md"The second response variable to investigate is `emotionality`"

# ╔═╡ 98bcdc51-cbba-409d-9fe2-6b9acf94c279
emotionality = Array{Float64}(dataset[:,7]);

# ╔═╡ ed50cd54-1a62-4474-88b8-0338023315d2
md"""For both response variables we can apply averaging/smoothing and plot the data to see if we can find long term effects. Adjust the settings below to see how averaging and smoothing help to see more general patterns.
"""

# ╔═╡ 47438d1b-b777-4909-9011-96b76a311544
begin
	checkWellbeing = @bind plotWellbeing CheckBox(default=true)
	checkEmotionality = @bind plotEmotionality CheckBox(default=false)
	
	md"""**Choose response variable(s) to display:**

	Plot wellbeing $checkWellbeing \
	Plot emotionality $checkEmotionality 
	"""
end

# ╔═╡ bc5a23a4-861c-4d26-a3c5-9c2b9eb76a2b
begin 
	timeSlider = @bind time Slider(1:100, show_value=true, default=30)
	smoothingSlider = @bind smoothing Slider(0:200, show_value=true, default=100)
	
	md"""**Set parameters:**

	Time frame to average over: $timeSlider days

	Degree of additional smoothing: $smoothingSlider"""
end

# ╔═╡ 0a54a5e1-d6dd-4fa4-b9d5-901c211250c9
md"""The plot with the default settings indicates a general trend where wellbeing tends to oscillate with a period of 2.5 months, or 5 oscillations per year. As a general trend, wellbeing also seems to increase within this data set. Also, when looking at the raw data without averaging or smoothing, it is visible that the variance appears to decrease over time which might pose some problems for the analysis.\
For emotionality on the other hand, patterns are less clear, but also decreasing variance is visible."""

# ╔═╡ b8b3a141-4e73-42ee-902c-296dc9938b72
md"These observations shows that there is at least some structure in the data that can be exploited for modelling. It indicates that including past measurement of the response variables might helpt the predictions."

# ╔═╡ b4c01050-247f-4b7a-a08c-aace0ba29b71
md"#### The predictor variables"

# ╔═╡ 321bc3e8-9d5e-439b-8051-9650e44d2268
predictor = dataset[:,11:end] # separate the predictor variables

# ╔═╡ d4f5845d-5f93-4128-b786-e8f48ca1bed3
md"Crop data set to remove data points with missing value for `exercise`"

# ╔═╡ f4ec23bb-423e-4253-9f27-11d79577fb1b
predictor_cropped = predictor[sum(predictor[:,"exercise"].===missing)+1:end,:];

# ╔═╡ 0ca328af-ae2a-4784-91ea-2abb4bda988a
md"Also crop the response variables to match the predictor data" 

# ╔═╡ 97fa1a98-6473-4317-b1bc-b1edf5e76504
md"## The analysis"

# ╔═╡ 4aecaa36-4ebc-4d80-8123-dd6a3087b87a
begin
	wellbeingBox = @bind responseVar Radio(["wellbeing","emotionality"], default="wellbeing")
	md"""**Choose response variable to analyse:**

	$wellbeingBox 
	"""
end

# ╔═╡ 5e93efdb-216a-4483-a222-d6398c3010c6
md"Split into a train and test set. The test set is here chosen to be 1/4 of the data."

# ╔═╡ 813e6e2d-39b1-4963-92ff-55f2b9f4c1a8
md"### Using genetic programming (SymbolicRegression)"

# ╔═╡ 808ecc62-13bb-4e1e-9307-aa5b5ea1e5d0
md"In this project, the default loss function of `EquationSearch` is used, being least squares. This makes it more comparable with classical linear regression."

# ╔═╡ 4d13d667-01b7-4dd9-abcf-6214b87c26db
options = SymbolicRegression.Options(
	# Choose the operators allowed to use in the equation
    binary_operators=(+, *, /, -, ^), 
    unary_operators=(tanh, relu),
	# Choose the number of "genetic" populations of equations to start with
    npopulations=6
);

# ╔═╡ 3d24ffe4-d5a7-4243-8649-44ef25ab65a2
md"Please note that the code block below is the bottleneck of the notebook in terms of runtime."

# ╔═╡ 68dae25b-76da-4ac3-8cc9-2e8a41ddf7ba
md"We investigate the one with the lowest MSE."

# ╔═╡ 5247b2c1-462e-43ae-80df-e46c2de360b3
md"Finally, we try out this equation to make the predictions."

# ╔═╡ cb63b0a3-b1c2-4115-924d-f6f4ce343fe8
begin
	trainTestBox = @bind trainTest Radio(["all","train","test"], default="all")
	md"""**Plot predictions for which part of the data?**

	$trainTestBox 
	"""
end

# ╔═╡ 1649eebb-3d07-4327-983f-32311fd0b730
begin 
	timeSlider2 = @bind time2 Slider(1:50, show_value=true, default=1)
	smoothingSlider2 = @bind smoothing2 Slider(0:100, show_value=true, default=0)
	
	md"""**Optional averaging/smoothing:**

	Time frame to average over: $timeSlider2 days

	Degree of additional smoothing: $smoothingSlider2"""
end

# ╔═╡ bc4db347-df99-4f82-9eb5-7deb234a3c37
md"Let's calculate the MSE for the train and test set. Notice that from the different dominating models, we take the \"best\" one, meaning the one with the lowest MSE on the train set."

# ╔═╡ adf4631c-e693-41a6-97cf-71aee3dc4d6a
md"As a test, we can take one of the dominating equations with fewer variables, instead of the one with the lowest MSE."

# ╔═╡ d3146239-e036-4629-9df3-0d8bdaf6d360
md"### Using linear regression"

# ╔═╡ e500e164-40df-4484-9bc9-9e2b03867076
md"Create a model that includes all variables."

# ╔═╡ db0ad624-122c-41e1-86b1-fc752d2f49fd
formula = @formula(response ~ wellbeingPast2 + emotionalityPast2 + meditationPast2 + exercisePast2 + sleepDurationPast2 + sleepEndPast2 + sleepDurationVarPast2 + sleepEndVarPast2 + wellbeingPast7 + emotionalityPast7 + meditationPast7 + exercisePast7 + sleepDurationPast7 + sleepEndPast7 + sleepDurationVarPast7 + sleepEndVarPast7 + wellbeingPast21 + emotionalityPast21 + meditationPast21 + exercisePast21 + sleepDurationPast21 + sleepEndPast21 + sleepDurationVarPast21 + sleepEndVarPast21 + wellbeingPast60 + emotionalityPast60 + meditationPast60 + exercisePast60 + sleepDurationPast60 + sleepEndPast60 + sleepDurationVarPast60 + sleepEndVarPast60);

# ╔═╡ 42eb0cd6-842c-44ba-b74f-7dab91bbc3d4
md"Train the model."

# ╔═╡ af7c2479-9e1c-4154-9b89-66f36c395fbf
md"Now let's again calculate the MSE on train and test set."

# ╔═╡ cd4b8636-98e6-475a-b44c-b542cd134fd1
md"""**Comparing linear regression to genetic programming**

For the data set used in this project, on the test set, some of the genetic programming equations seems to perform better than classical linear regression, particularly the equations with fewer variables included.\
A critical note here is that excluding some of the less relevant variables from the linear model, or using regularization as in ridge regression might help with this problem. This is however outside the scope of this project."""

# ╔═╡ 2c7a1446-94f6-4844-a8c3-2dc9af345592
md"""### Conclusions
For wellbeing, the genetic programming approach to regression works reasonably well. Something that is noticeble however is that the first part of the data tend to get a better fit that the later time points. This is probably because there is more variability in the earlier time points. Therefore, bad predictions in that part of the data result in a higher loss and are thus penalized harder.

Emotionality seems to be harder to predict. Here the formula heavily depends on past measurements of emotionality, rather than improving on that by using other variables.

When comparing `SymbolicRegression` to linear regression, the former can perform better, though not by a huge difference. Note however that the number of iterations and population size has been kept quite low to keep the runtime feasable. Increasing both might be needed to show the true potential of `SymbolicRegression` so that it clearly outperforms classical linear regression.
"""

# ╔═╡ df82d32c-c21c-4e57-a271-73de74bd71ab
md"## Appendix"

# ╔═╡ 13b7c0f0-d2ad-4582-be41-1e35858c2236
"""A function that centers and scales a matrix"""
scale(A) = (A .- mean(A,dims=1)) ./ std(A,dims=1)

# ╔═╡ 5814d9d8-8325-469b-b68e-1944ddbef429
"""A function that calculates the moving average of the last n days, either in- or excluding the current day"""
function n_days_average(dataPoints; n=10, onlyPast=false)
	shift = onlyPast ? 1 : 0
	return [(i < (n+shift) || dataPoints[i-n+1-shift] === missing) ? missing : sum(dataPoints[(i-n+1-shift):(i-shift)])/n for i in (n+shift):length(dataPoints)]
end

# ╔═╡ 9baaa0bf-d150-4b09-8a83-d4e64f407fba
"""A function that calculates variance of the last n days, either in- or excluding the current day"""
function n_days_variance(dataPoints; n=10, onlyPast=false)
	shift = onlyPast ? 1 : 0
	return [i < (n+shift) ? missing : var(dataPoints[(i-n+1-shift):(i-shift)]) for i in (n+shift):length(dataPoints)]
end

# ╔═╡ 284402aa-d794-4e29-b407-5674befbfc81
begin
	aggrPeriod = [2, 7, 21, 60] # Time frames to aggregate over
	varNames = [] # Names of the new variables
	nameBase = ["wellbeing", "emotionality", "meditation", "exercise", "sleepDuration", "sleepEnd", "sleepDurationVar", "sleepEndVar"]
	newVars = []
	for i in aggrPeriod
		push!(newVars, n_days_average(wellbeing, n=i, onlyPast=true))
		push!(newVars, n_days_average(emotionality, n=i, onlyPast=true))
		push!(newVars, n_days_average(predictor_cropped[:,"meditation"], n=i, 
              onlyPast=true))
		push!(newVars, n_days_average(predictor_cropped[:,"exercise"], n=i, 
              onlyPast=true))
		push!(newVars, n_days_average(predictor_cropped[:,"sleepDuration"], n=i, 
              onlyPast=true))
		push!(newVars, n_days_average(predictor_cropped[:,"sleepEnd"], n=i, 
              onlyPast=true))
		push!(newVars, n_days_variance(predictor_cropped[:,"sleepDuration"], n=i, 
              onlyPast=true))
		push!(newVars, n_days_variance(predictor_cropped[:,"sleepEnd"], n=i, 
              onlyPast=true))
		append!(varNames, nameBase.*("Past"*string(i)))
	end
	# Crop the new variables to be of the same size
	minSize = minimum(length(t) for t in newVars)
	newVars = [newVar[length(newVar)-minSize+1:end] for newVar in newVars]
	# Gather all in a matrix suitable for EquationSearch()
	X = [newVars[i][j] for i in 1:length(newVars), j in 1:length(newVars[1])]
	varNames = Vector{String}(varNames)
end

# ╔═╡ 0782f22f-abd5-4600-93d8-351b7e66e46a
md"Create derived variables by aggregating info on past data points. The average value of the past $n$ days is calculated. Also info on the past values of the resonse variables are included. Secondly, for sleepDuration and sleepEnd also the variance is calculated as a potentially informative variable. The time periods to aggregate over are here chosen to be $(string(aggrPeriod)) days."

# ╔═╡ 99a68eb2-c058-4224-95d8-83edc893ac5f
y_Wb = wellbeing[length(wellbeing)-size(X)[2]+1:end];

# ╔═╡ be3ca46a-9ac8-484d-a93f-84a6bfaea4ec
y_Em = emotionality[length(emotionality)-size(X)[2]+1:end];

# ╔═╡ 7a3521c6-7abb-4c6b-a782-2dc05f837e96
y = responseVar == "wellbeing" ? y_Wb : y_Em;

# ╔═╡ fe7be177-5e10-4f6c-af39-6442068bc2bc
begin
	k = 4
	X_test, X_train = X[:,1:k:end], X[:,Not(1:k:end)]
	y_test, y_train = y[1:k:end], y[Not(1:k:end)]
end;

# ╔═╡ d6c8df62-2afa-4f6f-9516-cc613a588fbe
hallOfFame = EquationSearch(X_train, y_train, niterations=4, varMap=varNames, options=options, numprocs=0);

# ╔═╡ 09c0929b-184c-4c03-b5cb-954d58a4eee6
dominating = calculateParetoFrontier(X_train, y_train, hallOfFame, options);

# ╔═╡ 573e8315-2b7f-4e61-8150-434c5dfb6fd9
best = argmin([dominating[i].score for i in 1:length(dominating)]);

# ╔═╡ 6c092a62-c9c3-4e9a-8da7-566098ed1c5d
begin
	isare = length(dominating)==1 ? "is" : "are"
	model = length(dominating)==1 ? "model" : "models"
	md"There $isare $(length(dominating)) dominating $model."
end

# ╔═╡ 4797578a-5007-4717-83f8-faf42fb2dcba
y_pred_train = evalTreeArray(dominating[best].tree, X_train, options)[1];

# ╔═╡ 88e3a5a9-4332-4936-aaa7-a20d3304e766
y_pred_test = evalTreeArray(dominating[best].tree, X_test, options)[1];

# ╔═╡ aa646ac8-8f66-4c51-be59-937869942f49
md"For the training set, the MSE is $(round(mean((y_train.-y_pred_train).^2),digits=4)). However, for the test set, this is $(round(mean((y_test.-y_pred_test).^2),digits=4)). So we see that there is some overfitting."

# ╔═╡ 148fdaa4-872c-474c-8303-98451087ad35
y_pred_train2 = evalTreeArray(dominating[1].tree, X_train, options)[1];

# ╔═╡ 25b36fc0-88a1-4508-a14f-e36807d0bc9c
y_pred_test2 = evalTreeArray(dominating[1].tree, X_test, options)[1];

# ╔═╡ df9e9389-3450-4ca1-b0cf-d75168e17c4f
md"Now for the training set, the MSE is $(round(mean((y_train.-y_pred_train2).^2),digits=4)) and for the test set it is $(round(mean((y_test.-y_pred_test2).^2),digits=4)). The amount of overfitting seems to be reduced and the performance on the test set is improved!"

# ╔═╡ a8091427-ac31-4f98-a347-c5ae15645f83
eqn = node_to_symbolic(dominating[best].tree, options, varMap=varNames);

# ╔═╡ 1ecf6a71-4db1-444f-ac08-fd8fa4c9b232
md"""The full formula goes as follows:

$eqn"""

# ╔═╡ 83562b00-f1a9-4e86-84c4-7a4a7c59ade5
begin
	# Convert matrix format to DataFrame
	X_df = DataFrame(X', :auto)
	rename!(X_df, names(X_df) .=> varNames)
	Xy_df = copy(X_df)
	Xy_df = insertcols!(Xy_df, 1, :response => y)
	# Split into train and test set
	X_df_test, X_df_train = X_df[1:k:end,:], X_df[Not(1:k:end),:]
	y_df_test, y_df_train = y[1:k:end], y[Not(1:k:end)]
	Xy_df_test, Xy_df_train = Xy_df[1:k:end,:], Xy_df[Not(1:k:end),:]
end;

# ╔═╡ 772db893-fa81-4396-96d2-a6559331ca76
linReg = lm(formula, Xy_df_train);

# ╔═╡ 0534dc12-569c-42fa-8d67-9d42aa76ca66
y_pred_linReg_train = predict(linReg, Xy_df_train);

# ╔═╡ 7fff987d-1c79-4597-95dc-f6621f09f2ae
y_pred_linReg_test = predict(linReg, Xy_df_test);

# ╔═╡ a0587b87-9bc3-4228-a64c-93df52d2cad8
md"For the training set, the MSE is $(round(mean((y_train.-y_pred_linReg_train).^2),digits=4)). For the test set, this is $(round(mean((y_test.-y_pred_linReg_test).^2),digits=4)). We conclude that there is quite some overfitting."

# ╔═╡ 595dfe24-379c-4fc3-9cfe-95a1b65d1710
"""A function that smooths data by averaging every two neighboring data points iteratively, with the number of iterations corresponding to the smoothness parameter"""
function smooth(dataPoints; smoothness=10)
	data = copy(dataPoints)
	for i in 1:smoothness
		data = n_days_average(data, n=2)
	end
	return data[2:end]
end

# ╔═╡ 5afac8d7-d2e1-4ec9-b644-7730a760c154
begin
	y1_sm = smooth(n_days_average(scale(wellbeing), n=time), smoothness=smoothing)
	y2_sm = smooth(n_days_average(scale(emotionality), n=time), smoothness=smoothing)
	title="Response variable(s): smoothed and averaged"
	if plotWellbeing && plotEmotionality
		plot(1:(length(y1_sm)),y1_sm, label="wellbeing", legend=:topleft, title=title, xlabel="Day", ylabel="Score")
		plot!(1:(length(y2_sm)),y2_sm, label="emotionality", legend=:topleft)
	elseif plotWellbeing
		plot(1:(length(y1_sm)),y1_sm, label="wellbeing", legend=:topleft, title=title, xlabel="Day", ylabel="Score")
	elseif plotEmotionality
		plot(1:(length(y2_sm)),y2_sm, label="emotionality", legend=:topleft, title=title)
	end
end

# ╔═╡ da0f15f3-a18b-41e3-97df-2b0589c72799
begin
	X_chosen = [X,X_train,X_test][trainTest .== ["all","train","test"]][1]
	y_chosen = [y,y_train,y_test][trainTest .== ["all","train","test"]][1]
	y_pred = evalTreeArray(dominating[best].tree, X_chosen, options)[1]
	y0 = smooth(n_days_average(y_chosen, n=time2), smoothness=smoothing2)
	y1 = smooth(n_days_average(y_pred, n=time2), smoothness=smoothing2)
	plot(1:length(y0), y0, label="original data", xlabel="Day", ylabel="Score")
	plot!(1:length(y0), y1, label="predictions")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
MultivariateStats = "6f286f6a-111f-5878-ab1e-185364afe411"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
SymbolicRegression = "8254be44-1295-4e6a-a16d-46603ac705cb"
SymbolicUtils = "d1185830-fcd6-423d-90d6-eec64667417b"

[compat]
CSV = "~0.9.11"
DataFrames = "~1.3.1"
GLM = "~1.6.1"
HTTP = "~0.9.17"
MultivariateStats = "~0.8.0"
Plots = "~0.29.9"
PlutoUI = "~0.7.1"
SymbolicRegression = "~0.4.11"
SymbolicUtils = "~0.6.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractAlgebra]]
deps = ["InteractiveUtils", "LinearAlgebra", "Markdown", "Random", "RandomExtensions", "SparseArrays", "Test"]
git-tree-sha1 = "7df2949bfd757e426897a4b579fbd5dc776ff8c9"
uuid = "c3fe647b-3220-5bb0-a1ea-a7954cac585d"
version = "0.12.0"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "265b06e2b1f6a216e0e8f183d28e4d354eab3220"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.2.1"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "49f14b6c56a2da47608fe30aed711b5882264d7a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.9.11"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "d711603452231bad418bd5e0c91f1abd650cba71"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.3"

[[ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "cfdfef912b7f93e4b848e80b9befdf9e331bc05a"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.1"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "9bc5dac3c8b6706b58ad5ce24cffd9861f07c94f"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.9.0"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "c6dd4a56078a7760c04b882d9d94a08a4669598d"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.44"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "c82bef6fc01e30d500f588cd01d29bdd44f1924e"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.3.0"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "04d13bfa8ef11720c24e4d840c0033d145537df7"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.17"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "502b3de6039d5b78c76118423858d981349f3823"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.9.7"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "8b3c09b56acaf3c0e581c66638b85c8650ee9dca"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.8.1"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "2b72a5624e289ee18256111657663721d59c143e"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.24"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "fb764dacfa30f948d52a6a4269ae293a479bbc62"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.6.1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "LinearAlgebra", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "7ea6f715b7caa10d7ee16f1cfcd12f3ccc74116a"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.48.0"

[[GeometryTypes]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "d796f7be0383b5416cd403420ce0af083b0f9b28"
uuid = "4d00f742-c7ba-57c2-abde-4428a4b178cb"
version = "0.8.5"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "8d70835a3759cdd75881426fced1508bb7b7e1b6"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.1"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[NaNMath]]
git-tree-sha1 = "f755f36b19a5116bb580de457cda0c140153f283"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.6"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "916077e0f0f8966eb0dc98a5c39921fdb8f49eb4"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.6.0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "ee26b350276c51697c9c2d88a072b339f9f03d73"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.5"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "d7fa6237da8004be601e19bd6666083056649918"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.3"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "87a4ea7f8c350d87d3a8ca9052663b633c0b2722"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "1.0.3"

[[PlotUtils]]
deps = ["Colors", "Dates", "Printf", "Random", "Reexport"]
git-tree-sha1 = "51e742162c97d35f714f9611619db6975e19384b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "0.6.5"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "FFMPEG", "FixedPointNumbers", "GR", "GeometryTypes", "JSON", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "Reexport", "Requires", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "f226ff9b8e391f6a10891563c370aae8beb5d792"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "0.29.9"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "Logging", "Markdown", "Random", "Suppressor"]
git-tree-sha1 = "45ce174d36d3931cd4e37a47f93e07d1455f038d"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.1"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "db3a23166af8aebf4db5ef87ac5b00d36eb771e2"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.0"

[[PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "2cf929d64681236a2e074ffafb8d568733d2e6af"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.3"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RandomExtensions]]
deps = ["Random", "SparseArrays"]
git-tree-sha1 = "062986376ce6d394b23d5d90f01d81426113a3c9"
uuid = "fb686558-2515-59ef-acaa-46db3789a887"
version = "0.4.3"

[[RecipesBase]]
git-tree-sha1 = "b4ed4a7f988ea2340017916f7c9e5d7560b52cae"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "0.8.0"

[[Reexport]]
deps = ["Pkg"]
git-tree-sha1 = "7b1d07f411bc8ddb7977ec7f377b97b158514fe0"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "0.2.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "8f82019e525f4d5c669692772a6f4b0a58b06a6a"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.2.0"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "244586bc07462d22aed0113af9c731f2a518c93e"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.10"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "ee010d8f103468309b8afac4abb9be2e18ff1182"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "0.3.2"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["OpenSpecFun_jll"]
git-tree-sha1 = "d8d8b8a9f4119829410ecd706da4cc8594a1e020"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "0.10.3"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "7f5a513baec6f122401abfc8e9c074fdac54f6c1"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.4.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "d88665adc9bcf45903013af0982e2fd05ae3d0a6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "51383f2d367eb3b444c961d485c565e4c0cf4ba0"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.14"

[[StatsFuns]]
deps = ["Rmath", "SpecialFunctions"]
git-tree-sha1 = "ced55fd4bae008a8ea12508314e725df61f0ba45"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.7"

[[StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "677488c295051568b0b79a77a8c44aa86e78b359"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.28"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[SymbolicRegression]]
deps = ["Distributed", "Optim", "Pkg", "Printf", "Random", "SpecialFunctions", "SymbolicUtils"]
git-tree-sha1 = "9c8754cff82a9faf13af4bf76b7503f3943fcf93"
uuid = "8254be44-1295-4e6a-a16d-46603ac705cb"
version = "0.4.11"

[[SymbolicUtils]]
deps = ["AbstractAlgebra", "Combinatorics", "DataStructures", "NaNMath", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "b486b44ca0fc12e713a819184b29f9b585e7ab7e"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "0.6.3"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "7cb456f358e8f9d102a8b25e8dfedf58fa5689bc"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.13"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "c69f9da3ff2f4f02e811c3323c22e5dfcb584cfa"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.1"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─ecfd3b68-0f66-4695-87a3-404eafad37bc
# ╟─94e88596-464d-41e7-a0da-68fc69f4e9c6
# ╟─d09fa8c4-c19d-4e5f-9d5f-c7831ec3762b
# ╠═5f9b5d30-68cb-11ec-2e1e-aff540c73252
# ╟─cc166935-f9a3-4b8e-b953-d97034e45dca
# ╠═4ec5675e-a40b-4f12-bffc-9c6aeeebf7c0
# ╟─ee67a2b5-8cb8-427f-9f38-286671c6b482
# ╟─5e762872-22a7-40aa-ae5a-441fdbcf462e
# ╠═13af064b-9abc-4c8e-9ada-831910ced6fb
# ╠═b001aa84-527b-4ccd-96aa-8c4c9434870a
# ╟─370172dc-15b3-490b-aa69-eb61d4272344
# ╟─1b159d85-f073-4c0a-ab88-1ba7b54eae3b
# ╟─2872be01-1202-4cbc-8fe0-19da723b5848
# ╟─1d0c8ca3-6eeb-4fcb-b813-112895aa7103
# ╠═e5327245-9eed-44a8-af67-aca7160ac3a9
# ╠═8ed46b21-3cd3-4f95-956f-0c1498273e11
# ╠═43a4f4a1-2fa3-4eaf-bbae-7c484d0a6ee0
# ╟─82a567c0-ee21-4e5c-a079-736cf5a9493a
# ╟─8e9f1328-8a8e-4629-8004-8efd31429b69
# ╠═98bcdc51-cbba-409d-9fe2-6b9acf94c279
# ╟─ed50cd54-1a62-4474-88b8-0338023315d2
# ╟─47438d1b-b777-4909-9011-96b76a311544
# ╟─bc5a23a4-861c-4d26-a3c5-9c2b9eb76a2b
# ╟─5afac8d7-d2e1-4ec9-b644-7730a760c154
# ╟─0a54a5e1-d6dd-4fa4-b9d5-901c211250c9
# ╟─b8b3a141-4e73-42ee-902c-296dc9938b72
# ╟─b4c01050-247f-4b7a-a08c-aace0ba29b71
# ╠═321bc3e8-9d5e-439b-8051-9650e44d2268
# ╟─d4f5845d-5f93-4128-b786-e8f48ca1bed3
# ╠═f4ec23bb-423e-4253-9f27-11d79577fb1b
# ╟─0782f22f-abd5-4600-93d8-351b7e66e46a
# ╠═284402aa-d794-4e29-b407-5674befbfc81
# ╟─0ca328af-ae2a-4784-91ea-2abb4bda988a
# ╠═99a68eb2-c058-4224-95d8-83edc893ac5f
# ╠═be3ca46a-9ac8-484d-a93f-84a6bfaea4ec
# ╟─97fa1a98-6473-4317-b1bc-b1edf5e76504
# ╟─4aecaa36-4ebc-4d80-8123-dd6a3087b87a
# ╟─7a3521c6-7abb-4c6b-a782-2dc05f837e96
# ╟─5e93efdb-216a-4483-a222-d6398c3010c6
# ╠═fe7be177-5e10-4f6c-af39-6442068bc2bc
# ╟─813e6e2d-39b1-4963-92ff-55f2b9f4c1a8
# ╟─808ecc62-13bb-4e1e-9307-aa5b5ea1e5d0
# ╠═4d13d667-01b7-4dd9-abcf-6214b87c26db
# ╟─3d24ffe4-d5a7-4243-8649-44ef25ab65a2
# ╠═d6c8df62-2afa-4f6f-9516-cc613a588fbe
# ╠═09c0929b-184c-4c03-b5cb-954d58a4eee6
# ╠═573e8315-2b7f-4e61-8150-434c5dfb6fd9
# ╟─6c092a62-c9c3-4e9a-8da7-566098ed1c5d
# ╟─68dae25b-76da-4ac3-8cc9-2e8a41ddf7ba
# ╠═a8091427-ac31-4f98-a347-c5ae15645f83
# ╟─1ecf6a71-4db1-444f-ac08-fd8fa4c9b232
# ╟─5247b2c1-462e-43ae-80df-e46c2de360b3
# ╟─cb63b0a3-b1c2-4115-924d-f6f4ce343fe8
# ╟─1649eebb-3d07-4327-983f-32311fd0b730
# ╟─da0f15f3-a18b-41e3-97df-2b0589c72799
# ╟─bc4db347-df99-4f82-9eb5-7deb234a3c37
# ╠═4797578a-5007-4717-83f8-faf42fb2dcba
# ╠═88e3a5a9-4332-4936-aaa7-a20d3304e766
# ╟─aa646ac8-8f66-4c51-be59-937869942f49
# ╟─adf4631c-e693-41a6-97cf-71aee3dc4d6a
# ╠═148fdaa4-872c-474c-8303-98451087ad35
# ╠═25b36fc0-88a1-4508-a14f-e36807d0bc9c
# ╟─df9e9389-3450-4ca1-b0cf-d75168e17c4f
# ╟─d3146239-e036-4629-9df3-0d8bdaf6d360
# ╠═83562b00-f1a9-4e86-84c4-7a4a7c59ade5
# ╟─e500e164-40df-4484-9bc9-9e2b03867076
# ╠═db0ad624-122c-41e1-86b1-fc752d2f49fd
# ╟─42eb0cd6-842c-44ba-b74f-7dab91bbc3d4
# ╠═772db893-fa81-4396-96d2-a6559331ca76
# ╟─af7c2479-9e1c-4154-9b89-66f36c395fbf
# ╠═0534dc12-569c-42fa-8d67-9d42aa76ca66
# ╠═7fff987d-1c79-4597-95dc-f6621f09f2ae
# ╟─a0587b87-9bc3-4228-a64c-93df52d2cad8
# ╟─cd4b8636-98e6-475a-b44c-b542cd134fd1
# ╟─2c7a1446-94f6-4844-a8c3-2dc9af345592
# ╟─df82d32c-c21c-4e57-a271-73de74bd71ab
# ╠═13b7c0f0-d2ad-4582-be41-1e35858c2236
# ╠═5814d9d8-8325-469b-b68e-1944ddbef429
# ╠═9baaa0bf-d150-4b09-8a83-d4e64f407fba
# ╠═595dfe24-379c-4fc3-9cfe-95a1b65d1710
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
