package sqlstore

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"
)

// encodeEmbedding serialises a float32 slice to compact binary.
// Each float32 is stored as 4 bytes in little-endian order.
func encodeEmbedding(v []float32) []byte {
	if len(v) == 0 {
		return nil
	}
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// decodeEmbedding reconstructs a float32 slice from its stored encoding.
// It handles both binary (4-byte aligned) and JSON array formats.
func decodeEmbedding(raw []byte) []float32 {
	if len(raw) == 0 {
		return nil
	}
	if len(raw)%4 == 0 && !looksLikeJSON(raw) {
		n := len(raw) / 4
		out := make([]float32, n)
		for i := range out {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out
	}
	var out []float32
	if json.Unmarshal(raw, &out) == nil {
		return out
	}
	return nil
}

func looksLikeJSON(b []byte) bool {
	return len(b) > 0 && b[0] == '['
}

// flexTime implements sql.Scanner for time columns that may be returned as
// time.Time (PostgreSQL, MySQL) or string (SQLite).
type flexTime struct {
	Time  time.Time
	Valid bool
}

func (ft *flexTime) Scan(src any) error {
	if src == nil {
		ft.Valid = false
		return nil
	}
	switch v := src.(type) {
	case time.Time:
		ft.Time = v
		ft.Valid = true
		return nil
	case string:
		// Strip Go's monotonic clock suffix (e.g. " m=+123.456")
		// which appears when time.Time.String() is stored directly.
		if idx := strings.Index(v, " m="); idx != -1 {
			v = v[:idx]
		}
		for _, layout := range []string{
			time.RFC3339Nano,
			time.RFC3339,
			"2006-01-02 15:04:05.999999999 -0700 MST",
			"2006-01-02 15:04:05.999999 -0700 MST",
			"2006-01-02 15:04:05 -0700 MST",
			"2006-01-02T15:04:05",
			"2006-01-02 15:04:05.999999999-07:00",
			"2006-01-02 15:04:05.999999999",
			"2006-01-02 15:04:05",
		} {
			if t, err := time.Parse(layout, v); err == nil {
				ft.Time = t
				ft.Valid = true
				return nil
			}
		}
		return fmt.Errorf("flexTime: cannot parse %q", v)
	default:
		return fmt.Errorf("flexTime: unsupported type %T", src)
	}
}
