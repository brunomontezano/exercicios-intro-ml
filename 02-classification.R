# Exercício de Regressão Logística #########################################
# Objetivo: fazer um modelo preditivo usando regressão logística para prever
# sobreviventes no desastre do Titanic (variável 'Survived').
#
# A meta é bater o resultado da tabela abaixo (no last_fit):
# A tibble: 2 x 3
#   .metric  .estimator .estimate
#   <chr>    <chr>      <dbl>
# 1 accuracy binary     0.802
# 2 roc_auc  binary     0.865

# PASSO 0) CARREGAR AS BASES -----------------------------------------------
# Instale o pacote titanic para pegar a base de dados "titanic::titanic_train"

# PASSO 1) BASE TREINO/TESTE -----------------------------------------------

set.seed(1)

titanic <- titanic::titanic_train |>
    dplyr::mutate(
        Survived = factor(
            Survived,
            levels = c(1, 0),
            labels = c("yes", "no")
            )
    )

titanic_initial_split <- rsample::initial_split(titanic)

titanic_train <- rsample::training(titanic_initial_split)
titanic_test <- rsample::testing(titanic_initial_split)

# PASSO 2) EXPLORAR A BASE ------------------------------------------

skimr::skim(titanic_train)

dplyr::glimpse(titanic_train)

# Problemas para analisar:
# 1) A proporção da variável resposta está balanceada?

titanic_train |>
    janitor::tabyl(Survived)

# Sim, existe um desbalanceamento sem necessidade de intervenção.

# 1) Valores faltantes: existem variáveis com poucos dados faltantes?
# Não, somente uma variável possui dado faltante (ver abaixo).

DataExplorer::plot_missing(titanic_train)

# 1) Valores faltantes: existem variáveis com muitos dados faltantes?
# É possível imputar?
# Sim, existem 129 valores ausentes na variável Age (idade).
# É possível imputar.

# 2) Os tipos das variáveis estão OK? Numéricas são numericas e categóricas são
# categóricas?
# ID está como numérico.

# 3) Existem variáveis categóricas com muitas categorias? (10 ou mais)
# Sim, Cabin, Ticket e Name.

titanic_train |>
    dplyr::select(where(is.character)) |>
    purrr::map(unique)

# 3) Existem variáveis categóricas com textos muito poluídos e com informação
# escondida a ser extraída?
# Sim, a variável Name tem nomes próprios. As variáveis Ticket.
# e Cabin são pouco informativas, além da variável Cabin possuir
# valores de string vazios.

# 4) Existem variáveis numéricas com muitos zeros?

colSums(titanic_train == 0, na.rm = TRUE) |>
    tibble::as_tibble(rownames = "variable") |>
    dplyr::rename(zero_count = value)

# Sim, as variáveis SibSp e Parch.

# 4) Existem variáveis constantes? (que é tudo igual na base inteira)

titanic_train |>
    purrr::map(unique) |>
    purrr::map(length)

# Não.

# 4) Existem variáveis numéricas muito assimétricas com potencial de outliers?

titanic_train |>
    dplyr::select(where(is.numeric)) |>
    tidyr::pivot_longer(
        cols = dplyr::everything(),
        names_to = "variables",
        values_to = "values"
    ) |>
    ggplot2::ggplot(ggplot2::aes(x = values)) +
    ggplot2::geom_histogram() +
    ggplot2::facet_wrap(~ variables, scales = "free")

# Fare e SibSp possuem outliers.

# 4) Existem variáveis numéricas com escalas muito diferentes (uma vai de -1 a
# 1 e a outra vai de 0 a 1000, por exemplo)?
# Sim, por exemplo as variáveis Fare e SibSp.

# 5) Existem variáveis explicativas muito correlacionadas umas com as outras?

titanic_train |>
    dplyr::select(where(is.numeric)) |>
    GGally::ggpairs()

# Aparentemente não.

# 6) 'Dummificar' variáveis categóricas
# (exceto a variável resposta,'outcome')
# (importantíssimo!!)

# Yeaaaah.

# PASSO 3) DATAPREP --------------------------------------------------
titanic_recipe <- recipes::recipe(
    Survived ~ .,
    data = titanic_train
    ) |>
  recipes::step_zv(recipes::all_predictors()) |>
  recipes::step_rm(Name, Ticket, Cabin) |>
  recipes::step_impute_mode(recipes::all_nominal_predictors()) |>
  recipes::step_impute_median(recipes::all_numeric_predictors()) |>
  recipes::step_novel(recipes::all_nominal_predictors()) |>
  recipes::step_dummy(recipes::all_nominal_predictors())

# Para ficar checando o resultado do recipe

  titanic_recipe |>
    recipes::prep() |>
    recipes::bake(new_data = NULL) |>
    dplyr::glimpse()

# PASSO 4) MODELO -----------------------------------------------------
# f(x), modo e engine
# Tunar penalty e deixar mixture = 1

mod <- parsnip::logistic_reg(
    penalty = tune::tune(),
    mixture = 1
    ) |>
    parsnip::set_engine("glmnet") |>
    parsnip::set_mode("classification")

# PASSO 5) WORKFLOW ----------------------------------------------------
# workflow add_model add_recipe

wflow <- workflows::workflow() |>
    workflows::add_recipe(titanic_recipe) |>
    workflows::add_model(mod)

# PASSO 6) TUNAGEM DE HIPERPARÂMETROS -----------------------------------
# vfold_cv() tune_grid()

cv <- rsample::vfold_cv(
    data = titanic_train,
    v = 5,
    strata = "Survived"
    )

res <- tune::tune_grid(
    object = wflow,
    resamples = cv,
    grid = dials::grid_regular(
                dials::penalty(
                    range = c(-4, -1)),
                levels = 20
                ),
    metrics = yardstick::metric_set(
        yardstick::roc_auc,
        yardstick::accuracy),
    control = tune::control_grid(verbose = TRUE, allow_par = TRUE)
)

# PASSO 7) SELECAO DO MODELO FINAL ---------------------------------------
# autoplot select_best finalize_workflow last_fit

ggplot2::autoplot(res)

best_hp <- res |>
    tune::select_best(metric = "roc_auc")

wflow <- wflow |>
    tune::finalize_workflow(best_hp)

titanic_last_fit <- tune::last_fit(
    object = wflow,
    split = titanic_initial_split
    )

# PASSO 8) DESEMPENHO DO MODELO FINAL -----------------------------------
# collect_metrics, collect_predictions, roc_curve, autoplot

titanic_last_fit |>
    tune::collect_metrics()

titanic_last_fit |>
    tune::collect_predictions()

titanic_last_fit |>
    tune::collect_predictions() |>
    yardstick::roc_curve(.pred_yes, truth = Survived) |>
    ggplot2::autoplot()

# PASSO 9) Variáveis importantes ---------------------------------------

# Plot
vip::vip(titanic_last_fit$.workflow[[1]]$fit$fit$fit)

# Tibble
imp_tbl <- vip::vi(titanic_last_fit$.workflow[[1]]$fit$fit$fit)

imp_tbl

# PASSO 10) MODELO FINAL ----------------------------------------------
# fit (usando a base titanic)

titanic_final <- wflow |> fit(titanic)

# PASSO 11) GUARDA TUDO -----------------------------------------------
# 1) Guarda o last_fit
# 2) Guarda o fit
# 3) Guarda o que quiser reutilizar depois, por exemplo a tabela do vi().

readr::write_rds(
    x = titanic_last_fit,
    file = "/tmp/titanic_last_fit.rds"
)

readr::write_rds(
    x = titanic_final,
    file = "/tmp/titanic_final_model.rds"
)

readr::write_rds(
    x = imp_tbl,
    file = "/tmp/titanic_imp_tbl.rds"
)

# PASSO 12) ESCORE A BASE DE TESTE DO KAGGLE ---------------------------

titanic_final |>
    predict(new_data = titanic::titanic_test, type = "prob")
