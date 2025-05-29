1. Certifique-se de que o Optuna já foi instalado:
   Para instalar o Optuna, utilize o comando:
   pip install optuna

2. Crie e configure o estudo:
   O estudo já foi criado e está em andamento. Se o estudo foi configurado corretamente com study.optimize, você pode agora acessar o dashboard.

3. Iniciar o Dashboard:
   Para visualizar os resultados do seu estudo no dashboard, basta rodar o seguinte comando no terminal:
   optuna-dashboard "sqlite:///CAMINHO/arquivo_estudo.db" ------> uma dica aqui e copiar o caminho absoluto do arquivo e colar depois das 3 barras apos sqlite
   - sqlite:///optuna_study.db: Aqui você deve indicar o banco de dados onde os resultados do estudo estão sendo armazenados. Caso o estudo tenha sido salvo em outro banco, substitua o caminho para o banco correto.
   - O comando iniciará um servidor web local que você poderá acessar no seu navegador.

4. Acessar o Dashboard:
   Após rodar o comando, você verá a seguinte mensagem no terminal:
   Dashboards available at:
   - http://127.0.0.1:5006/
   Isso significa que você pode acessar o dashboard no navegador, entrando no endereço http://127.0.0.1:5006/.

5. Explorar o Dashboard:
   No dashboard, você poderá visualizar e analisar os resultados do seu estudo. Aqui estão as principais funcionalidades:
   - Visualização de Resultados: O dashboard exibe uma tabela com as tentativas realizadas no estudo. Você pode ver os parâmetros sugeridos e o valor da função objetivo para cada tentativa.
   - Visualização Gráfica: Há gráficos interativos que mostram a evolução das tentativas, como a evolução do valor da função objetivo ao longo das tentativas e a relação entre os parâmetros.
   - Comparação de Tentativas: O Optuna permite comparar diferentes tentativas de uma forma visualmente acessível.

6. Interação com o Dashboard:
   - Parâmetros: Você pode observar como os diferentes parâmetros do estudo (por exemplo, x, y, etc.) afetam o resultado da função objetivo.
   - Métricas: O dashboard também fornece uma visão geral das métricas relacionadas à otimização.
   - Análises Avançadas: É possível gerar gráficos como:
     - Distribuição de Parâmetros: Veja a distribuição dos valores dos parâmetros durante as tentativas.
     - Importância dos Parâmetros: Veja como cada parâmetro contribui para o desempenho do modelo.

7. Finalizando o Estudo:
   Após acompanhar os resultados, você pode decidir parar ou finalizar o estudo. Caso precise salvar o estudo para visualizações futuras, ele já estará automaticamente armazenado no banco de dados especificado (optuna_study.db no exemplo acima).

8. Desligar o Dashboard:
   Para parar o servidor do dashboard, basta pressionar Ctrl + C no terminal onde o comando foi executado.






###########################################

#Limpa banco de dados para trials sujos de teste

#import sqlite3

# Conecta no banco
conn = sqlite3.connect("optuna_study.db")
cursor = conn.cursor()

# Apaga todos os trials
cursor.execute("DELETE FROM trials")

# Apaga todos os estudos
cursor.execute("DELETE FROM studies")

# Salva as mudanças
conn.commit()

conn.close()

print("Banco limpo! Todas as tabelas foram esvaziadas.")

