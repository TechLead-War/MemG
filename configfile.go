package memg

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// FileConfig represents the YAML/JSON config file structure.
// Field names use json tags for both JSON and simple key parsing.
type FileConfig struct {
	Port   int    `json:"port"`
	Target string `json:"target"`
	Entity string `json:"entity"`
	DB     string `json:"db"`

	LLM struct {
		Provider string `json:"provider"`
		Model    string `json:"model"`
		APIKey   string `json:"api_key"`
		BaseURL  string `json:"base_url"`
	} `json:"llm"`

	Embed struct {
		Provider string `json:"provider"`
		Model    string `json:"model"`
		APIKey   string `json:"api_key"`
		BaseURL  string `json:"base_url"`
	} `json:"embed"`

	Recall struct {
		Limit            int     `json:"limit"`
		Threshold        float64 `json:"threshold"`
		SummaryLimit     int     `json:"summary_limit"`
		SummaryThreshold float64 `json:"summary_threshold"`
	} `json:"recall"`

	Session struct {
		Timeout string `json:"timeout"`
	} `json:"session"`

	WorkingMemory struct {
		Turns int `json:"turns"`
	} `json:"working_memory"`
	Memory struct {
		TokenBudget   int `json:"token_budget"`
		SummaryBudget int `json:"summary_budget"`
	} `json:"memory"`

	Conscious        *bool  `json:"conscious"`
	ConsciousLimit   int    `json:"conscious_limit"`
	ConsciousCacheTTL string `json:"conscious_cache_ttl"`
	PruneInterval    string `json:"prune_interval"`
	Debug            bool   `json:"debug"`
}

// DefaultConfigPaths returns the paths searched for a config file, in priority order.
func DefaultConfigPaths() []string {
	paths := []string{
		"memg.json",
		"memg.config.json",
	}

	home, err := os.UserHomeDir()
	if err == nil {
		paths = append(paths,
			filepath.Join(home, ".memg", "config.json"),
		)
	}

	return paths
}

// LoadConfigFile searches for a config file in the default paths and loads
// the first one found. Returns an empty FileConfig (not an error) if no file exists.
func LoadConfigFile() (*FileConfig, string, error) {
	return LoadConfigFileFrom(DefaultConfigPaths())
}

// LoadConfigFileFrom tries each path in order and loads the first one found.
// Returns (config, path-used, error). If no file exists, returns a zero FileConfig
// with empty path and nil error.
func LoadConfigFileFrom(paths []string) (*FileConfig, string, error) {
	for _, p := range paths {
		expanded := expandTilde(p)
		data, err := os.ReadFile(expanded)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, "", fmt.Errorf("read config %s: %w", expanded, err)
		}

		var fc FileConfig
		if err := json.Unmarshal(data, &fc); err != nil {
			return nil, expanded, fmt.Errorf("parse config %s: %w", expanded, err)
		}
		return &fc, expanded, nil
	}

	return &FileConfig{}, "", nil
}

func (fc *FileConfig) ProxyPort(def int) int {
	if fc.Port > 0 {
		return fc.Port
	}
	return def
}

func (fc *FileConfig) ProxyTarget(def string) string {
	if fc.Target != "" {
		return fc.Target
	}
	return def
}

func (fc *FileConfig) ProxyEntity(def string) string {
	if fc.Entity != "" {
		return fc.Entity
	}
	return def
}

func (fc *FileConfig) ProxyDB(def string) string {
	if fc.DB != "" {
		return fc.DB
	}
	return def
}

func (fc *FileConfig) LLMProviderName(def string) string {
	if fc.LLM.Provider != "" {
		return fc.LLM.Provider
	}
	return def
}

func (fc *FileConfig) LLMModelName(def string) string {
	if fc.LLM.Model != "" {
		return fc.LLM.Model
	}
	return def
}

func (fc *FileConfig) EmbedProviderName(def string) string {
	if fc.Embed.Provider != "" {
		return fc.Embed.Provider
	}
	return def
}

func (fc *FileConfig) EmbedModelName(def string) string {
	if fc.Embed.Model != "" {
		return fc.Embed.Model
	}
	return def
}

func (fc *FileConfig) LLMBaseURL(def string) string {
	if fc.LLM.BaseURL != "" {
		return fc.LLM.BaseURL
	}
	return def
}

func (fc *FileConfig) EmbedBaseURL(def string) string {
	if fc.Embed.BaseURL != "" {
		return fc.Embed.BaseURL
	}
	return def
}

func (fc *FileConfig) RecallLimit(def int) int {
	if fc.Recall.Limit > 0 {
		return fc.Recall.Limit
	}
	return def
}

func (fc *FileConfig) RecallThresholdVal(def float64) float64 {
	if fc.Recall.Threshold > 0 {
		return fc.Recall.Threshold
	}
	return def
}

func (fc *FileConfig) RecallSummaryLimitVal(def int) int {
	if fc.Recall.SummaryLimit > 0 {
		return fc.Recall.SummaryLimit
	}
	return def
}

func (fc *FileConfig) RecallSummaryThresholdVal(def float64) float64 {
	if fc.Recall.SummaryThreshold > 0 {
		return fc.Recall.SummaryThreshold
	}
	return def
}

func (fc *FileConfig) SessionTimeoutDuration(def time.Duration) time.Duration {
	if fc.Session.Timeout != "" {
		if d, err := time.ParseDuration(fc.Session.Timeout); err == nil {
			return d
		}
	}
	return def
}

func (fc *FileConfig) PruneIntervalDuration(def time.Duration) time.Duration {
	if fc.PruneInterval != "" {
		if d, err := time.ParseDuration(fc.PruneInterval); err == nil {
			return d
		}
	}
	return def
}

func (fc *FileConfig) ConsciousMode(def bool) bool {
	if fc.Conscious != nil {
		return *fc.Conscious
	}
	return def
}

func (fc *FileConfig) ConsciousLimitVal(def int) int {
	if fc.ConsciousLimit > 0 {
		return fc.ConsciousLimit
	}
	return def
}

func (fc *FileConfig) ConsciousCacheTTLDuration(def time.Duration) time.Duration {
	if fc.ConsciousCacheTTL != "" {
		if d, err := time.ParseDuration(fc.ConsciousCacheTTL); err == nil {
			return d
		}
	}
	return def
}

func (fc *FileConfig) WorkingMemoryTurns(def int) int {
	if fc.WorkingMemory.Turns > 0 {
		return fc.WorkingMemory.Turns
	}
	return def
}

func (fc *FileConfig) MemoryTokenBudget(def int) int {
	if fc.Memory.TokenBudget > 0 {
		return fc.Memory.TokenBudget
	}
	return def
}

func (fc *FileConfig) SummaryTokenBudget(def int) int {
	if fc.Memory.SummaryBudget > 0 {
		return fc.Memory.SummaryBudget
	}
	return def
}

func expandTilde(path string) string {
	if len(path) == 0 || path[0] != '~' {
		return path
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return path
	}
	return filepath.Join(home, path[1:])
}
