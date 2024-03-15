import os
from typing import List

import oci
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())


def embed_documents(texts: List[str]) -> List[List[float]]:
    # デフォルトの場所にあるDEFAULTプロファイルを使用してデフォルトの設定を作成します。
    # 詳細については、
    # https://docs.cloud.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm#SDK_and_CLI_Configuration_File
    # を参照してください。
    config = oci.config.from_file(file_location=os.environ["OCI_CONFIG_FILE"], profile_name="CHICAGO")

    # デフォルト構成ファイルを使用してサービスクライアントを初期化します。
    generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        config)
    # サービスにリクエストを送信します。
    embed_text_response = generative_ai_inference_client.embed_text(
        embed_text_details=oci.generative_ai_inference.models.EmbedTextDetails(
            inputs=texts,
            serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
                model_id="cohere.embed-multilingual-v3.0"),
            compartment_id=os.environ["OCI_COMPARTMENT_OCID"],
            input_type="SEARCH_DOCUMENT"))

    # レスポンスからデータを取得します。
    embeddings = embed_text_response.data.embeddings
    return [list(map(float, e)) for e in embeddings]
