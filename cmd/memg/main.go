package main

import (
	"fmt"
	"os"

	"memg"
)

const version = "0.1.0"

func main() {
	memg.LoadEnv()

	if len(os.Args) < 2 {
		printUsage()
		return
	}

	switch os.Args[1] {
	case "proxy":
		runProxy(os.Args[2:])
	case "mcp":
		runMCP(os.Args[2:])
	case "bench":
		runBench(os.Args[2:])
	case "version":
		fmt.Printf("memg v%s\n", version)
	case "help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("MemG - Pluggable memory layer for language model applications")
	fmt.Println()
	fmt.Println("Usage:")
	fmt.Println("  memg <command>")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  proxy      Start the memory-augmenting reverse proxy")
	fmt.Println("  mcp        Start the MCP (Model Context Protocol) server")
	fmt.Println("  bench      Run the LoCoMo memory benchmark")
	fmt.Println("  version    Print the current version")
	fmt.Println("  help       Show this help message")
}
