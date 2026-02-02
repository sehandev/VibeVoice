"""
VibeVoice-ASR Modal Deployment

Modal 함수로 B200 GPU를 사용해 VibeVoice-ASR 추론을 실행합니다.

Usage:
    # 단일 파일 추론
    modal run modal_asr.py --audio-file /path/to/audio.wav
"""

import modal

# =============================================================================
# Generation 설정 파라미터
# =============================================================================

# GPU 설정
GPU_TYPE = "H200"  # H100, L40S 등 선택 가능 (B200은 현재 컨테이너에서 미지원)

# 생성 파라미터
MAX_NEW_TOKENS = 32768       # 생성할 최대 토큰 수 (긴 오디오용)

# Beam Search 설정 (num_beams > 1이면 beam search 사용)
NUM_BEAMS = 4                # 빔 개수 (높을수록 정확도가 향상되나 속도가 느려짐. 성능 중시는 5~10 권장)

# Sampling 설정 (num_beams == 1 이고 temperature > 0 일 때 사용)
TEMPERATURE = 0.0            # 샘플링 온도 (Beam Search 사용 시 무시됨)
TOP_P = 1.0                  # nucleus sampling 확률 임계값 (Beam Search 사용 시 무시됨)

# =============================================================================

# Modal App 정의
app = modal.App("vibevoice-asr")

# 컨테이너 이미지 정의 (NVIDIA PyTorch 베이스 이미지 사용)
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:24.07-py3",
        add_python="3.11",
    )
    .apt_install(
        "ffmpeg",
        "libsndfile1",
        "clang",      # flash-attn 빌드에 필요
        "build-essential",
    )
    .pip_install(
        "wheel",
        "setuptools",
        "packaging",
        "ninja",  # flash-attn 빌드 가속
    )
    .pip_install(
        "transformers>=4.51.3,<5.0.0",
        "accelerate",
        "llvmlite>=0.40.0",
        "numba>=0.57.0",
        "librosa",
        "huggingface_hub",
    )
    .run_commands(
        # flash-attn GPU 환경에서 빌드 (--no-build-isolation 필수)
        "pip install flash-attn --no-build-isolation",
        gpu=GPU_TYPE,
    )
    .run_commands(
        # VibeVoice 설치
        "pip install git+https://github.com/microsoft/VibeVoice.git",
    )
)

# 모델을 영구 저장할 Volume
model_volume = modal.Volume.from_name("vibevoice-asr-model", create_if_missing=True)
MODEL_DIR = "/models"


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    timeout=00,
    volumes={MODEL_DIR: model_volume},
)
class VibeVoiceASR:
    """VibeVoice-ASR 추론 클래스"""

    model_name: str = "microsoft/VibeVoice-ASR"

    @modal.enter()
    def load_model(self):
        """컨테이너 시작 시 모델 로드"""
        import os
        import torch
        from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
        from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

        model_path = os.path.join(MODEL_DIR, self.model_name.replace("/", "_"))

        # 모델이 없으면 다운로드
        if not os.path.exists(model_path) or not os.listdir(model_path):
            from huggingface_hub import snapshot_download
            print(f"모델 다운로드 중: {self.model_name}")
            snapshot_download(
                repo_id=self.model_name,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )
            model_volume.commit()

        print(f"모델 로드 중: {model_path}")

        # Processor 로드
        self.processor = VibeVoiceASRProcessor.from_pretrained(
            model_path,
            language_model_pretrained_name="Qwen/Qwen2.5-7B"
        )

        # 모델 로드 (flash_attention_2 사용)
        self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )
        self.model.eval()

        print("모델 로드 완료!")

    @modal.method()
    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
    ) -> dict:
        """
        오디오 파일을 텍스트로 변환합니다.

        Args:
            audio_bytes: 오디오 파일의 바이트 데이터
            filename: 파일 이름 (확장자로 형식 판단)

        Returns:
            dict: {
                "raw_text": 원본 텍스트,
                "segments": 구조화된 세그먼트 리스트,
                "generation_time": 생성 시간(초)
            }
        """
        import tempfile
        import os
        import time
        import torch

        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            print(f"오디오 처리 중: {filename}")

            # 오디오 처리
            inputs = self.processor(
                audio=[temp_path],
                sampling_rate=None,
                return_tensors="pt",
                padding=True,
                add_generation_prompt=True
            )

            # GPU로 이동
            inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}

            print(f"Input shape: {inputs['input_ids'].shape}")

            # Generation config (reference 구현과 동일한 로직)
            gen_config = {
                "max_new_tokens": MAX_NEW_TOKENS,
                "pad_token_id": self.processor.pad_id,
                "eos_token_id": self.processor.tokenizer.eos_token_id,
            }

            # Beam search vs sampling (reference 로직과 동일)
            if NUM_BEAMS > 1:
                gen_config["num_beams"] = NUM_BEAMS
                gen_config["do_sample"] = False
                print(f"Beam Search: num_beams={NUM_BEAMS}")
            else:
                do_sample = TEMPERATURE > 0
                gen_config["do_sample"] = do_sample
                if do_sample:
                    gen_config["temperature"] = TEMPERATURE
                    gen_config["top_p"] = TOP_P
                    print(f"Sampling: temp={TEMPERATURE}, top_p={TOP_P}")
                else:
                    print("Greedy Decoding")

            # 추론
            start_time = time.time()

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_config)

            generation_time = time.time() - start_time

            # 디코딩
            input_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[0, input_length:]

            # EOS 토큰 처리
            eos_positions = (generated_ids == self.processor.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                generated_ids = generated_ids[:eos_positions[0] + 1]

            raw_text = self.processor.decode(generated_ids, skip_special_tokens=True)

            # 구조화된 출력 파싱
            try:
                segments = self.processor.post_process_transcription(raw_text)
            except Exception as e:
                print(f"구조화 파싱 실패: {e}")
                segments = []

            print(f"추론 완료: {generation_time:.2f}초")

            return {
                "raw_text": raw_text,
                "segments": segments,
                "generation_time": generation_time,
                "num_beams": NUM_BEAMS,
                "gpu_type": GPU_TYPE,
            }

        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@app.local_entrypoint()
def main(audio_file: str = None):
    """로컬에서 실행할 엔트리포인트"""
    import json

    if not audio_file:
        print("사용법: modal run modal_asr.py --audio-file /path/to/audio.wav")
        return

    # ASR 인스턴스 생성
    asr = VibeVoiceASR()

    file_path = audio_file
    print(f"\n{'='*60}")
    print(f"처리 중: {file_path}")
    print('='*60)

    # 파일 읽기
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    # 추론 실행
    result = asr.transcribe.remote(
        audio_bytes=audio_bytes,
        filename=file_path,
    )

    # 결과 출력
    print(f"\n생성 시간: {result['generation_time']:.2f}초")
    print(f"\n--- Raw Output ---")
    print(result['raw_text'][:1000] + "..." if len(result['raw_text']) > 1000 else result['raw_text'])

    if result['segments']:
        print(f"\n--- Segments ({len(result['segments'])}개) ---")
        for seg in result['segments'][:10]:
            print(f"[{seg.get('start_time', 'N/A')} - {seg.get('end_time', 'N/A')}] "
                  f"Speaker {seg.get('speaker_id', 'N/A')}: {seg.get('text', '')}")
        if len(result['segments']) > 10:
            print(f"  ... 외 {len(result['segments']) - 10}개 세그먼트")

    # JSON 저장
    output_path = file_path.rsplit(".", 1)[0] + "_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장됨: {output_path}")
