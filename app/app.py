{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMjk66YKmA3lwyylCjU4akJ",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Mehnaz-Bashir/Human-Activity-Recognition/blob/main/app/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Foht3QqeSixy",
        "outputId": "4a451011-7eba-40e2-a544-f51f23e46c95"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing app.py\n"
          ]
        }
      ],
      "source": [
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import joblib\n",
        "\n",
        "# Load model\n",
        "model = joblib.load(\"../models/best_har_model.pkl\")\n",
        "feature_names = joblib.load(\"../models/feature_names.pkl\")\n",
        "\n",
        "st.title(\"Human Activity Recognition\")\n",
        "\n",
        "st.write(\n",
        "    \"Upload a CSV file containing sensor features to predict the activity.\"\n",
        ")\n",
        "\n",
        "uploaded_file = st.file_uploader(\n",
        "    \"Choose CSV file\",\n",
        "    type=[\"csv\"]\n",
        ")\n",
        "\n",
        "activity_map = {\n",
        "    1: \"WALKING\",\n",
        "    2: \"WALKING_UPSTAIRS\",\n",
        "    3: \"WALKING_DOWNSTAIRS\",\n",
        "    4: \"SITTING\",\n",
        "    5: \"STANDING\",\n",
        "    6: \"LAYING\"\n",
        "}\n",
        "\n",
        "if uploaded_file is not None:\n",
        "\n",
        "    data = pd.read_csv(uploaded_file)\n",
        "\n",
        "    st.subheader(\"Uploaded Data\")\n",
        "    st.dataframe(data.head())\n",
        "\n",
        "    predictions = model.predict(data)\n",
        "\n",
        "    predicted_labels = [\n",
        "        activity_map[p] for p in predictions\n",
        "    ]\n",
        "\n",
        "    data[\"Predicted Activity\"] = predicted_labels\n",
        "\n",
        "    st.subheader(\"Prediction Results\")\n",
        "    st.dataframe(data)\n",
        "\n",
        "    csv = data.to_csv(index=False).encode(\"utf-8\")\n",
        "\n",
        "    st.download_button(\n",
        "        label=\"Download Predictions\",\n",
        "        data=csv,\n",
        "        file_name=\"predictions.csv\",\n",
        "        mime=\"text/csv\"\n",
        "    )"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "os.listdir()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "rDZ6awweS2fO",
        "outputId": "09d89f46-168b-4137-92f7-63da457c0c3a"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['.config', 'app.py', 'feature_names.pkl', 'best_har_model.pkl', 'sample_data']"
            ]
          },
          "metadata": {},
          "execution_count": 3
        }
      ]
    }
  ]
}