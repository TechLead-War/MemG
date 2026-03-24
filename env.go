package memg

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
)

// LoadEnv loads environment variables from a .env file. It searches the
// current directory first, then ~/.memg/.env. Variables already set in
// the environment are never overwritten.
//
// A missing .env file is not an error — the function silently returns nil.
func LoadEnv() error {
	paths := []string{".env"}

	home, err := os.UserHomeDir()
	if err == nil {
		paths = append(paths, filepath.Join(home, ".memg", ".env"))
	}

	for _, p := range paths {
		if err := loadEnvFile(p); err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return err
		}
		return nil
	}
	return nil
}

func loadEnvFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		if line == "" || line[0] == '#' {
			continue
		}

		key, val, ok := parseEnvLine(line)
		if !ok {
			continue
		}

		if _, exists := os.LookupEnv(key); exists {
			continue
		}
		os.Setenv(key, val)
	}
	return scanner.Err()
}

func parseEnvLine(line string) (key, val string, ok bool) {
	// Strip inline comments only if unquoted.
	eq := strings.IndexByte(line, '=')
	if eq < 1 {
		return "", "", false
	}

	key = strings.TrimSpace(line[:eq])
	val = strings.TrimSpace(line[eq+1:])

	// Strip surrounding quotes.
	if len(val) >= 2 {
		if (val[0] == '"' && val[len(val)-1] == '"') ||
			(val[0] == '\'' && val[len(val)-1] == '\'') {
			val = val[1 : len(val)-1]
		}
	}

	return key, val, true
}
