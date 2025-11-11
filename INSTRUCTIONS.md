A atividade inicial:

```
Olá, pessoal 😃

Conforme conversamos na aula de ontem, segue atividade para a aula do dia 04/09.

Objetivo:
Pesquisar artigos no CVPR 2025 e escolher três para fundamentar um projeto futuro. Você pode trabalhar individualmente ou em dupla  (graduação + pós-graduação).

Requisitos:
1. Acesse o repositório da conferência: https://openaccess.thecvf.com/CVPR2025
2. Escolha 3 artigos que mais lhe interessa.
3. Monte um slide (Powerpoint ou Google Slides) para cada artigo contendo:
Título, autores e link de cada artigo.
Resumo ou descrição rápida (1–2 frases).
Justificativa da escolha.
4. Faça o upload do slide nessa atividade até a aula do dia 04/09/2025.
5. Se prepare para uma apresentação de até 10 minutos, que será realizada na aula do dia 04/09.

Importante: 
Se optar por duplas, elas devem ser formadas entre um aluno de graduação e um de pós-graduação.

Avaliação:
Será selecionado um dos três artigos para desenvolver o projeto ao longo do curso, com apresentação final ao término da disciplina.
```

Passo a passo do artigo:
0. Texto-para-imagem (Feito no artigo, mas vamos reproduzir o processo de geração da imagem a partir do texto)
1. Gerar normal maps a partir da imagem (Vamos reproduzir o processo de geração dos normal maps a partir da imagem)
2. Mesh Initialization com LRM ou Sphere init (IntantsMesh) (Vamos reproduzir o processo de inicialização da malha)
3. Mesh Refinement (3d enhancement com controlnet-tile e controlnet-normal + texto) (Vamos reproduzir o processo de refinamento da malha)
4. 
Objetivo final: Gerar uma malha 3D a partir de uma ou mais imagens, onde primeiro um texto descritivo da cena será gerado utilizando LLM para gerar o texto, extremamente detalhado. E outras imagens como o normal map e afins serão usadas de forma que a malha seja refinada de acordo com o texto e com essas técnicas.

Dataset que será usado: https://app.gazebosim.org/GoogleResearch (Vamos usar o dataset do Google Research)

Codebase: https://github.com/EnVision-Research/Kiss3DGen (Vamos usar o codebase do Kiss3DGen como base para o nosso projeto, mas vamos fazer algumas modificações para atender aos nossos objetivos e não copiaremos o código)