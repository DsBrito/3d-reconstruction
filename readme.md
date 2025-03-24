# Reconstrução 3D em Sala de Aula

## Descrição

Este repositório contém o código e os materiais necessários para a reconstrução 3D de uma caixa a partir de imagens estereoscópicas. O objetivo é calcular um mapa de disparidade, reconstruir as coordenadas tridimensionais da cena e gerar uma visualização colorida do modelo reconstruído.

## Estrutura do Repositório

- `esquerda.ppm` e `direita.ppm`: Imagens da câmera esquerda e direita, já retificadas.
- `box_data.py`: Dados de calibração das câmeras e informações sobre o deslocamento entre elas.
- `depth_map.py` e `depth_map.ipynb`: Rotinas para cálculo do mapa de disparidade usando Correlação Cruzada Normalizada (com ou sem filtro Gaussiano).
- `sparse_reconstruction01.py` e `sparse_reconstruction01.ipynb`: Exemplo de seleção de cores dos pixels para a reconstrução.
- `Result01.png`, `Result02.png`, `Result03.png`: Exemplos de resultados esperados.

## Objetivos do Trabalho

1. Calcular e exibir o mapa de disparidade entre as imagens retificadas.
2. Reconstruir as coordenadas tridimensionais (X, Y, Z) utilizando a metodologia abordada na Aula 11.
3. Gerar uma visualização 3D colorida, atribuindo a cada ponto sua cor correspondente na imagem original.
4. Opcionalmente, realizar a reconstrução de uma região específica da imagem utilizando uma ROI (Region of Interest).

## Como Executar

1.  Instale as dependências necessárias:
    ```bash
    pip install scipy
    ```
          <!-- 2. Execute o código de cálculo de disparidade:
             ```bash
             python main.py
             ```
       <!-- 2. Realize a reconstrução 3D:
          ```bash
          python sparse_reconstruction01.py
          ```
    <!-- 2. Visualize os resultados gerados. -->

## Observações

- As imagens de exemplo (`Result01.png`, `Result02.png`, `Result03.png`) servem apenas para comparação e não devem ser utilizadas diretamente na reconstrução.
- Para seleção de uma ROI, utilize a função `cv2.selectROI(image)` do OpenCV.
