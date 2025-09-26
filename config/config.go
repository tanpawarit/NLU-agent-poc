package config

import "github.com/kelseyhightower/envconfig"

func New[T any](prefix string) (*T, error) {
	var conf T
	if err := envconfig.Process(prefix, &conf); err != nil {
		return nil, err
	}

	return &conf, nil
}

func MustNew[T any](prefix string) *T {
	conf, err := New[T](prefix)
	if err != nil {
		panic(err)
	}
	return conf
}
