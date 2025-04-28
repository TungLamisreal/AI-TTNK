package main

/*
#cgo CFLAGS: -I./AI_SPK/whisper.cpp/include -I./AI_SPK/whisper.cpp/ggml/include
#cgo LDFLAGS: -L./AI_SPK/whisper.cpp/build/bin/Release -lwhisper
#include <stdlib.h>
#include <stdio.h>
#include "whisper.h"

// Helper function in C to get segment text safely
const char* get_segment_text(struct whisper_context *ctx, int index) {
    return whisper_full_get_segment_text(ctx, index);
}
*/
import "C"

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"

	//"os/exec" //cmt
	"time"
	"unsafe"
)

func loadWavFile(path string) ([]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %s: %w", path, err)
	}
	defer file.Close()

	header := make([]byte, 44)
	if _, err := file.Read(header); err != nil {
		return nil, fmt.Errorf("failed to read WAV header: %w", err)
	}

	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return nil, fmt.Errorf("invalid WAV format: missing RIFF/WAVE identifiers")
	}

	audioFormat := binary.LittleEndian.Uint16(header[20:22])
	if audioFormat != 1 {
		return nil, fmt.Errorf("unsupported WAV audio format: %d (expected 1 for PCM)", audioFormat)
	}

	bitsPerSample := binary.LittleEndian.Uint16(header[34:36])
	if bitsPerSample != 16 {
		return nil, fmt.Errorf("unsupported WAV bits per sample: %d (expected 16)", bitsPerSample)
	}

	numChannels := binary.LittleEndian.Uint16(header[22:24])
	if numChannels != 1 {
		fmt.Printf("Warning: WAV file has %d channels, processing as mono.\n", numChannels)
	}

	var samples []int16
	for {
		var sample int16
		err := binary.Read(file, binary.LittleEndian, &sample)
		if err != nil {
			break
		}
		samples = append(samples, sample)
	}

	if len(samples) == 0 {
		return nil, fmt.Errorf("no audio samples found in WAV file")
	}

	audio := make([]float32, len(samples))
	for i, s := range samples {
		audio[i] = float32(s) / 32768.0
	}

	fmt.Printf("Loaded %d audio samples from %s\n", len(audio), path)
	return audio, nil
}

// func RecordAudioWithFFMPEG(outputPath string) error {
// 	fmt.Println("Starting audio recording with ffmpeg...")
// 	cmd := exec.Command(
// 		"ffmpeg",
// 		"-y",
// 		"-f", "dshow",
// 		"-i", "audio=Microphone Array (Realtek(R) Audio)",
// 		"-t", "5",
// 		"-ar", "16000",
// 		"-ac", "1",
// 		"-c:a", "pcm_s16le",
// 		"-filter:a", "volume=3",
// 		outputPath,
// 	)

// 	cmd.Stdout = os.Stdout
// 	cmd.Stderr = os.Stderr

// 	return cmd.Run()
// }

func main() {
	modelPathStr := "AI_SPK/whisper.cpp/models/ggml-base.bin"
	wavPathStr := "nhacvn.wav"

	//cmt
	// Record audio
	// if err := RecordAudioWithFFMPEG(wavPathStr); err != nil {
	// 	fmt.Fprintf(os.Stderr, "Error: Failed to record audio: %v\n", err)
	// 	os.Exit(1)
	// }
	// fmt.Println("Audio recording complete.")

	time.Sleep(1 * time.Second)

	modelPath := C.CString(modelPathStr)
	defer C.free(unsafe.Pointer(modelPath))

	params := C.struct_whisper_context_params{}
	ctx := C.whisper_init_from_file_with_params(modelPath, params)
	if ctx == nil {
		fmt.Fprintf(os.Stderr, "Error: Failed to initialize whisper context from model '%s'\n", modelPathStr)
		os.Exit(1)
	}
	defer C.whisper_free(ctx)

	fmt.Printf("Successfully initialized whisper context from model '%s'\n", modelPathStr)

	audio, err := loadWavFile(wavPathStr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: Failed to load audio file '%s': %v\n", wavPathStr, err)
		os.Exit(1)
	}

	if len(audio) == 0 {
		fmt.Fprintf(os.Stderr, "Error: No audio data loaded from '%s'\n", wavPathStr)
		os.Exit(1)
	}

	fullParams := C.whisper_full_default_params(C.WHISPER_SAMPLING_GREEDY)
	fullParams.print_progress = C.bool(false)
	fullParams.print_special = C.bool(false)
	fullParams.print_realtime = C.bool(false)
	fullParams.print_timestamps = C.bool(false)
	fullParams.translate = C.bool(false)
	fullParams.language = nil
	fullParams.detect_language = C.bool(true)

	fmt.Println("Running transcription...")

	ret := C.whisper_full(ctx, fullParams, (*C.float)(unsafe.Pointer(&audio[0])), C.int(len(audio)))
	if ret != 0 {
		fmt.Fprintf(os.Stderr, "Error: Failed to run whisper_full, return code: %d\n", ret)
		os.Exit(1)
	}

	langId := C.whisper_full_lang_id(ctx)
	langStr := C.GoString(C.whisper_lang_str(langId))
	var langNameVN string
	switch langStr {
	case "af":
		langNameVN = "Tiếng Afrikaans"
	case "am":
		langNameVN = "Tiếng Amharic"
	case "ar":
		langNameVN = "Tiếng Ả Rập"
	case "as":
		langNameVN = "Tiếng Assam"
	case "az":
		langNameVN = "Tiếng Azerbaijan"
	case "ba":
		langNameVN = "Tiếng Bashkir"
	case "be":
		langNameVN = "Tiếng Belarus"
	case "bg":
		langNameVN = "Tiếng Bulgaria"
	case "bn":
		langNameVN = "Tiếng Bengal"
	case "bo":
		langNameVN = "Tiếng Tây Tạng"
	case "br":
		langNameVN = "Tiếng Breton"
	case "bs":
		langNameVN = "Tiếng Bosnia"
	case "ca":
		langNameVN = "Tiếng Catalan"
	case "cs":
		langNameVN = "Tiếng Séc"
	case "cy":
		langNameVN = "Tiếng Wales"
	case "da":
		langNameVN = "Tiếng Đan Mạch"
	case "de":
		langNameVN = "Tiếng Đức"
	case "el":
		langNameVN = "Tiếng Hy Lạp"
	case "en":
		langNameVN = "Tiếng Anh"
	case "es":
		langNameVN = "Tiếng Tây Ban Nha"
	case "et":
		langNameVN = "Tiếng Estonia"
	case "eu":
		langNameVN = "Tiếng Basque"
	case "fa":
		langNameVN = "Tiếng Ba Tư"
	case "fi":
		langNameVN = "Tiếng Phần Lan"
	case "fo":
		langNameVN = "Tiếng Faroe"
	case "fr":
		langNameVN = "Tiếng Pháp"
	case "gl":
		langNameVN = "Tiếng Galicia"
	case "gu":
		langNameVN = "Tiếng Gujarat"
	case "haw":
		langNameVN = "Tiếng Hawaii"
	case "he":
		langNameVN = "Tiếng Do Thái"
	case "hi":
		langNameVN = "Tiếng Hindi"
	case "hr":
		langNameVN = "Tiếng Croatia"
	case "ht":
		langNameVN = "Tiếng Haiti"
	case "hu":
		langNameVN = "Tiếng Hungary"
	case "hy":
		langNameVN = "Tiếng Armenia"
	case "id":
		langNameVN = "Tiếng Indonesia"
	case "is":
		langNameVN = "Tiếng Iceland"
	case "it":
		langNameVN = "Tiếng Ý"
	case "ja":
		langNameVN = "Tiếng Nhật"
	case "jw":
		langNameVN = "Tiếng Java"
	case "ka":
		langNameVN = "Tiếng Gruzia"
	case "kk":
		langNameVN = "Tiếng Kazakh"
	case "km":
		langNameVN = "Tiếng Khmer"
	case "kn":
		langNameVN = "Tiếng Kannada"
	case "ko":
		langNameVN = "Tiếng Hàn"
	case "la":
		langNameVN = "Tiếng Latin"
	case "lb":
		langNameVN = "Tiếng Luxembourg"
	case "ln":
		langNameVN = "Tiếng Lingala"
	case "lo":
		langNameVN = "Tiếng Lào"
	case "lt":
		langNameVN = "Tiếng Litva"
	case "lv":
		langNameVN = "Tiếng Latvia"
	case "mg":
		langNameVN = "Tiếng Malagasy"
	case "mi":
		langNameVN = "Tiếng Maori"
	case "mk":
		langNameVN = "Tiếng Macedonia"
	case "ml":
		langNameVN = "Tiếng Malayalam"
	case "mn":
		langNameVN = "Tiếng Mông Cổ"
	case "mr":
		langNameVN = "Tiếng Marathi"
	case "ms":
		langNameVN = "Tiếng Mã Lai"
	case "mt":
		langNameVN = "Tiếng Malta"
	case "my":
		langNameVN = "Tiếng Miến Điện"
	case "ne":
		langNameVN = "Tiếng Nepal"
	case "nl":
		langNameVN = "Tiếng Hà Lan"
	case "no":
		langNameVN = "Tiếng Na Uy"
	case "oc":
		langNameVN = "Tiếng Occitan"
	case "pa":
		langNameVN = "Tiếng Punjab"
	case "pl":
		langNameVN = "Tiếng Ba Lan"
	case "ps":
		langNameVN = "Tiếng Pashto"
	case "pt":
		langNameVN = "Tiếng Bồ Đào Nha"
	case "ro":
		langNameVN = "Tiếng Romania"
	case "ru":
		langNameVN = "Tiếng Nga"
	case "sa":
		langNameVN = "Tiếng Phạn"
	case "sd":
		langNameVN = "Tiếng Sindhi"
	case "si":
		langNameVN = "Tiếng Sinhala"
	case "sk":
		langNameVN = "Tiếng Slovakia"
	case "sl":
		langNameVN = "Tiếng Slovenia"
	case "sn":
		langNameVN = "Tiếng Shona"
	case "so":
		langNameVN = "Tiếng Somali"
	case "sq":
		langNameVN = "Tiếng Albania"
	case "sr":
		langNameVN = "Tiếng Serbia"
	case "su":
		langNameVN = "Tiếng Sundan"
	case "sv":
		langNameVN = "Tiếng Thụy Điển"
	case "sw":
		langNameVN = "Tiếng Swahili"
	case "ta":
		langNameVN = "Tiếng Tamil"
	case "te":
		langNameVN = "Tiếng Telugu"
	case "tg":
		langNameVN = "Tiếng Tajik"
	case "th":
		langNameVN = "Tiếng Thái"
	case "tk":
		langNameVN = "Tiếng Turkmen"
	case "tl":
		langNameVN = "Tiếng Tagalog"
	case "tr":
		langNameVN = "Tiếng Thổ Nhĩ Kỳ"
	case "tt":
		langNameVN = "Tiếng Tatar"
	case "uk":
		langNameVN = "Tiếng Ukraina"
	case "ur":
		langNameVN = "Tiếng Urdu"
	case "uz":
		langNameVN = "Tiếng Uzbek"
	case "vi":
		langNameVN = "Tiếng Việt"
	case "yi":
		langNameVN = "Tiếng Yiddish"
	case "zh":
		langNameVN = "Tiếng Trung"
	case "zu":
		langNameVN = "Tiếng Zulu"
	default:
		langNameVN = "Ngôn ngữ không xác định"
	}

	fmt.Printf("Ngôn ngữ phát hiện: %s (%s)\n", langNameVN, langStr)

	nSegments := int(C.whisper_full_n_segments(ctx))
	fmt.Printf("Number of text segments: %d\n", nSegments)

	var resultBuffer bytes.Buffer
	for i := 0; i < nSegments; i++ {
		cText := C.get_segment_text(ctx, C.int(i))
		if cText == nil {
			fmt.Fprintf(os.Stderr, "Warning: Failed to get text for segment %d\n", i)
			continue
		}
		goText := C.GoString(cText)
		resultBuffer.WriteString(goText + " ")
	}

	fmt.Println("\n--- Full Transcription ---")
	fmt.Println(resultBuffer.String())
	fmt.Println("--------------------------")
}
