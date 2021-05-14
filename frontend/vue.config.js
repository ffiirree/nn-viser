module.exports = {
    publicPath: process.env.NODE_ENV === 'production' ? '/static/web' : '',
    outputDir: '../static/web',
    assetsDir: '',
    productionSourceMap: false,
    devServer: {
        port: 8083,
        disableHostCheck: true,
        proxy: {
            '/api': {
                target: 'http://localhost:5000',
                pathRewrite: { '^/api': '' },
                ws: true,
                changeOrigin: true,
                secure: false, // https
            },
        }
    },
    parallel: require('os').cpus().length > 1,
    transpileDependencies: [
        /\bvue-awesome\b/
    ],
    chainWebpack: (config) => {
        if (process.env.NODE_ENV === 'production') {
            if (process.env.npm_config_report) {
                config
                    .plugin('webpack-bundle-analyzer')
                    .use(require('webpack-bundle-analyzer').BundleAnalyzerPlugin)
                    .end();

                config.plugins.delete('prefetch')
            }
        }
    }
};
