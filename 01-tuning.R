# A base de dados 'concrete' consiste em dados sobre a composição de diferentes
# misturas de concreto e a sua resistência à compressão - 'compressive_strength'
# (https://pt.wikipedia.org/wiki/Esfor%C3%A7o_de_compress%C3%A3o)
# O nosso objetivo é prever a resitência à compressão a partir dos ingredientes.
# A coluna 'compressive_strength' é a variável resposta. A coluna 'age' nos diz
# a idade do concreto na hora do teste (o concreto fica mais forte ao longo do
# tempo) e o resto das colunas como 'cement' e 'water' são componentes do
# concreto em kilogramas por metro cúbico.

# Pacotes -----------------------------------------------------------------


# Base de dados -----------------------------------------------------------

set.seed(666)

data(concrete, package = "modeldata")

dplyr::glimpse(concrete)

concrete |>
    tidyr::pivot_longer(
        cols = c(dplyr::everything(), - compressive_strength),
        names_to = "features",
        values_to = "values"
    ) |>
    ggplot2::ggplot(ggplot2::aes(y = compressive_strength, x = values)) +
    ggplot2::geom_point(size = 0.5) +
    ggplot2::geom_smooth(method = "lm", color = "royalblue") +
    ggplot2::facet_wrap(~ features, scales = "free_x")

# exercício 0 -------------------------------------------------------------
# Defina uma 'recipe' que normalize todas as variáveis explicativas.
# Dicas: recipe(), step_normalize(), all_numeric_predictors().

# Divisão entre base de treino e teste
concrete_initial_split <- rsample::initial_split(
                          data = concrete,
                          prop = 0.75,
                          )

concrete_train <- rsample::training(concrete_initial_split)
concrete_test <- rsample::testing(concrete_initial_split)

# Criação da receita
concrete_rec <- recipes::recipe(
                    data = concrete_train,
                    formula = compressive_strength ~ .) |>
                recipes::step_normalize(recipes::all_numeric_predictors())

# Base de treino após pré-processamento
concrete_rec |>
    recipes::prep(concrete_train) |>
    recipes::juice()

# exercício 1 -------------------------------------------------------------
# Defina uma especificação de f que caracterize uma regressão linear
# (mode 'regression'). Especifique também que você deseja tunar a 'penalty' e
# a 'mixture'.
# Dicas: linear_reg(), set_engine(), set_mode(), tune().

concrete_mod <- parsnip::linear_reg(
                    penalty = tune::tune(),
                    mixture = tune::tune()
                ) |>
                parsnip::set_engine("glmnet") |>
                parsnip::set_mode("regression")
                
# exercício 2 -------------------------------------------------------------
# Defina um 'workflow' que junte a receita do ex. 0 e o modelo do ex. 1.
# Dicas: workflow(), add_model(), add_recipe().

concrete_wf <- workflows::workflow() |>
    workflows::add_recipe(concrete_rec) |>
    workflows::add_model(concrete_mod)

# exercício 3 -------------------------------------------------------------
# Crie um objeto que represente a estratégia de reamostragem do tipo K-Fold
# cross validation com 5 folds.
# Dica: vfold_cv().

concrete_cv <- rsample::vfold_cv(
    data = concrete_train,
    v = 5
    )

# exercício 4 -------------------------------------------------------------
# Defina um grid de hiperparâmetros que você irá testar tanto de 'penalty'
# quanto
# de 'mixture'.
# Dica: grid_regular(), penalty(), mixture().

concrete_grid <- tidyr::crossing(
    penalty = seq(0.01, 1, length = 10),
    mixture = seq(0, 1, length = 5)
    )

# exercício 5 -------------------------------------------------------------
# Execute a tunagem do modelo usando os objetos criados nos exercícios
# anteriores.
# Dica: tune_grid().

concrete_tune <- tune::tune_grid(
                    object = concrete_wf,
                    resamples = concrete_cv,
                    grid = concrete_grid,
                    metrics = yardstick::metric_set(
                        yardstick::rmse,
                        yardstick::rsq,
                        yardstick::mae),
                    control = tune::control_grid(verbose = TRUE)
                    )

# exercício 6 -------------------------------------------------------------
# Visualize os resultados dos modelos ajustados e atualize o workflow com os
# parâmetros do melhor modelo.
# Dica: autoplot(), collect_metrics(), show_best(), select_best(),
# finalize_workflow().

concrete_tune |>
    ggplot2::autoplot()

concrete_tune |>
    tune::collect_metrics()

concrete_tune |>
    tune::show_best(metric = "rsq", n = 10)

concrete_best <- concrete_tune |>
    tune::select_best(metric = "rmse")

concrete_final_wf <- concrete_wf |>
    tune::finalize_workflow(concrete_best)

# desafio 1 ---------------------------------------------------------------
# Qual hiperparâmetro tem maior efeito no resultado do modelo? Justifique
# a sua afirmativa com um gráfico.

# Lambda.

concrete_tune |>
    tune::collect_metrics() |>
    dplyr::filter(.metric == "rmse") |>
    ggplot2::ggplot(
        ggplot2::aes(x = mixture, y = mean, color = as.factor(penalty))
    ) +
    ggplot2::geom_line(size = 2, alpha = 0.5) +
    ggplot2::labs(x = "Alfa", y = "Erro quadrático médio") +
    ggplot2::scale_color_discrete(name = "Lambda")

# exercício 7 -------------------------------------------------------------
# Ajuste o modelo na base de treino e verifique o desempenho na base de teste.
# Dica: last_fit(split = ______initial_split), collect_metrics()

concrete_last_fit <- concrete_final_wf |>
    tune::last_fit(split = concrete_initial_split)

concrete_last_fit |>
    tune::collect_metrics()

tune::collect_predictions(concrete_last_fit) %>%
  ggplot2::ggplot(ggplot2::aes(.pred, compressive_strength)) +
  ggplot2::geom_point() +
  ggplot2::geom_abline(
      intercept = 0, slope = 1, color = "red", size = 2, alpha = 0.5
  )

# exercício 8 -------------------------------------------------------------
# Ajuste o modelo final para a base inteira salve-o em um arquivo .rds.
# Dica: fit(), saveRDS().

concrete_final_model <- concrete_final_wf  |>
    parsnip::fit(data = concrete)

(comparison_predict_truth <- predict(
    object = concrete_final_model,
    new_data = concrete) |>
    dplyr::bind_cols(concrete$compressive_strength) |>
    purrr::set_names(c("predito", "real")))

readr::write_rds(
    x = concrete_final_model,
    file = "/tmp/concrete_final_model.rds")
