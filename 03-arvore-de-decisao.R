# Exercício de Árvore de Decisão #########################################

# Objetivo: refazer o modelo feito no exercício de regressão logística para
# prever sobreviventes no desastre do Titanic (variável 'Survived') usando
# Árvore de Decisão.

# Questões interessantes:
# Valeu a pena mudar de regressão logística para árvore de decisão? Justifique.
# As variáveis importantes são as mesmas em ambos os modelos?

# Dicas:
# 1) Tente reaproveitar ao máximo o código usado na regressão logística.

# 2) Identifique quais partes do código devem mudar por causa da troca de
#    logistic_reg() para decision_tree().

# Basicamente, a parte do pré-processamento. Por conta das diferenças na
# hora de manejar os dados dentro do algoritmo computacional. Além disso,
# deve-se atentar ao maior número de hiperparâmetros a serem tunados, o que
# pode acabar sendo dificultado na implantação de modelos de árvore em amostras
# menores.

# 3) Tome conhecimento sobre quais datapreps não precisa mais fazer
#    (scale, log, interação, etc).

# Não há necessidade de "dummificação" das variáveis, e a parte de imputação
# de dados também pode ser pulada considerando outros aspectos e a
# especificidade da demanda.

# 4) Crie códigos para comparar os dois modelos.
#    (duas curvas roc no mesmo gráfico)

# Setar seed
set.seed(1)

# Padronizar desfecho
titanic <- titanic::titanic_train |>
    dplyr::mutate(
        Survived = factor(
            Survived,
            levels = c(1, 0),
            labels = c("yes", "no")
            )
    )

# Separar base
titanic_initial_split <- rsample::initial_split(titanic)

titanic_train <- rsample::training(titanic_initial_split)
titanic_test <- rsample::testing(titanic_initial_split)

# Criar pré-processamento para modelo de regressão logística
lr_rec <- recipes::recipe(
    Survived ~ .,
    data = titanic_train
    ) |>
  recipes::step_zv(recipes::all_predictors()) |>
  recipes::step_rm(Name, Ticket, Cabin) |>
  recipes::step_impute_mode(recipes::all_nominal_predictors()) |>
  recipes::step_impute_median(recipes::all_numeric_predictors()) |>
  recipes::step_novel(recipes::all_nominal_predictors()) |>
  recipes::step_dummy(recipes::all_nominal_predictors())

# Criar pré-processamento para modelo de árvore de decisão
dt_rec <- recipes::recipe(
    Survived ~ .,
    data = titanic_train
    ) |>
  recipes::step_rm(Name, Ticket, Cabin) |>
  recipes::step_novel(recipes::all_nominal_predictors()) |>
  recipes::step_zv(recipes::all_predictors())

# Criar especificação da regressão LASSO
lr_spec <- parsnip::logistic_reg(
    penalty = tune::tune(),
    mixture = 1
    ) |>
    parsnip::set_engine("glmnet") |>
    parsnip::set_mode("classification")

# Criar especificação da árvore de decisão
dt_spec <- parsnip::decision_tree(
    cost_complexity = tune::tune(),
    tree_depth = tune::tune(),
    min_n = tune::tune()
    ) |>
    parsnip::set_engine("rpart") |>
    parsnip::set_mode("classification")

# Criar workflow set
tit_models <- workflowsets::workflow_set(
    preproc = list(linear = lr_rec, cart = dt_rec),
    models = list(glmnet = lr_spec, decision_tree = dt_spec),
    cross = FALSE
    )

# Criar objeto para cross-validation
splits <- rsample::vfold_cv(
    data = titanic_train,
    v = 5,
    repeats = 1,
    strata = Survived)

# Rodar cross-validation
tit_models <-
   tit_models |> workflowsets::workflow_map(
       fn = "tune_grid",
       resamples = splits,
       grid = 10,
       metrics = yardstick::metric_set(
           yardstick::accuracy,
           yardstick::roc_auc
       ),
       verbose = TRUE,
       control = tune::control_grid(
           save_pred = TRUE,
           save_workflow = TRUE),
       seed = 1)

# Checar AUC e acurácia de ambos os modelos
ggplot2::autoplot(tit_models)

# Comparar modelos através da AUC
workflowsets::rank_results(
    tit_models,
    rank_metric = "roc_auc",
    select_best = TRUE) |>
   dplyr::filter(.metric == "roc_auc") |>
   dplyr::select(model, .config, auc = mean, rank)

# Verificar os 10 folds de cada um dos algoritmos (comparando AUC)
tit_models |>
    tune::collect_metrics() |>
    dplyr::filter(.metric == "roc_auc") |>
    dplyr::arrange(dplyr::desc(mean))

# Comparar melhor modelo de cada workflow
ggplot2::autoplot(
   tit_models,
   rank_metric = "roc_auc",  # <- Como ordenar os modelos
   metric = "roc_auc",       # <- Métrica a ser visualizada
   select_best = TRUE     # <- Um ponto por workflow
) +
   ggplot2::geom_text(
       ggplot2::aes(y = 0.9, label = wflow_id),
       angle = 90,
       hjust = 1) +
   ggplot2::labs(y = "AUC", x = "Ranking dos workflows") +
   ggplot2::theme(legend.position = "none")

# Tuning dos parâmetros do modelo de regressão logística
ggplot2::autoplot(
    tit_models,
    id = "linear_glmnet",
    metric = "roc_auc")

# Tuning dos parâmetros do modelo de árvore de decisão
ggplot2::autoplot(
    tit_models,
    id = "cart_decision_tree",
    metric = "roc_auc")

# Selecionar o modelo de árvore para finalizar e verificar parâmetros
(best_results <-
   tit_models |>
   workflowsets::extract_workflow_set_result("cart_decision_tree") |>
   tune::select_best(metric = "roc_auc"))

# Realizar último ajuste no split inicial na base do Titanic
# com melhor modelo de árvore de decisão baseado na AUC
dt_results <-
   tit_models |>
   workflowsets::extract_workflow("cart_decision_tree") |>
   workflows::finalize_workflow(best_results) |>
   tune::last_fit(split = titanic_initial_split)

# Métricas da árvore de decisão na base de testes
dt_results |>
    tune::collect_metrics()

# Histograma das predições nos sobreviventes na base de teste
dt_results |>
   tune::collect_predictions() |>
   dplyr::filter(Survived == "yes") |>
   dplyr::count(.pred_yes) |>
   ggplot2::ggplot(ggplot2::aes(x = .pred_yes, y = n)) +
   ggplot2::geom_col(fill = "royalblue") +
   ggplot2::scale_x_continuous(labels = scales::percent_format()) +
   ggplot2::labs(
       x = "Probabilidade predita",
       y = "Número de sobreviventes preditos com determinada probabilidade"
   )
