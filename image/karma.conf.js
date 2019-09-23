process.env.CHROME_BIN = require('puppeteer').executablePath();

module.exports = function(config) {
  config.set({
    frameworks: ['mocha', 'karma-typescript'],//, 'benchmark'],
    files: [
        {pattern: 'src/**/*.ts'},
        {pattern: 'test/**/*.ts'},
        {pattern: 'test/**/*.html'},
        {pattern: 'dist/**/*.min.js'}
    ],
    preprocessors: {
      '**/*.ts': ['karma-typescript'],  // *.tsx for React Jsx
    },
    karmaTypescriptConfig: {
      tsconfig: 'tsconfig.json',
      // we include this here to keep our test files separate,
      // but not include tests in normal builds
      include: [ 'test/**/*.ts'],
      reports: {} // Do not produce coverage html.
    },
    // logLevel: config.LOG_DEBUG,
    reporters: ['progress', 'karma-typescript'],//, 'benchmark'],
    browsers: ['Chrome_no_sandbox'],
    customLaunchers: {
      Chrome_no_sandbox: {
        base: 'Chrome',
        flags: [
          '--no-sandbox',
          '--disable-setuid-sandbox',
          '--headless',
          '--disable-gpu',
          '--remote-debugging-port=9222',
        ],
      },
    },
    reportSlowerThan: 500,
    browserNoActivityTimeout: 500000,
    browserDisconnectTimeout: 300000,
    pingTimeout: 1000000
  });
};
